//! Safe MCP-to-NATS adapter.
//!
//! This module accepts explicit MCP JSON-RPC metadata and tool-call arguments only.
//! Hidden chain-of-thought/interleaved reasoning is not exposed by standard MCP
//! messages and is intentionally neither parsed nor routed. Any non-schema field
//! named as reasoning is rejected rather than treated as executable input.

use async_nats::{Client, Message};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::{collections::BTreeMap, sync::Arc};
use thiserror::Error;
use tokio::sync::Mutex;

const GENERATE_SUBJECT: &str = "glm.evoke.generate";
const ATTEST_SUBJECT: &str = "glm.evoke.attest";
const SEAL_SUBJECT: &str = "glm.evoke.seal";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolContract {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub input_schema: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractSet {
    pub tools: Vec<ToolContract>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RpcEnvelope {
    pub jsonrpc: String,
    pub id: Value,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInvocation {
    pub tool: String,
    #[serde(default)]
    pub arguments: Value,
    pub agent_role: String,
    pub trace_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub contract: ToolContract,
    pub invocation: ToolInvocation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttestRequest {
    pub tool: String,
    pub agent_role: String,
    pub arguments: Value,
    pub trace_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SealRequest {
    pub trace_id: String,
    pub tool: String,
    pub state: TraceState,
    #[serde(default)]
    pub metadata: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TraceState {
    ContractExtracted,
    Attested,
    ModelRefuted,
    Published,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attestation {
    pub allowed: bool,
    pub reason: String,
}

#[derive(Debug, Error)]
pub enum AdapterError {
    #[error("invalid MCP envelope: {0}")]
    InvalidEnvelope(String),
    #[error("unsupported MCP method: {0}")]
    UnsupportedMethod(String),
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    #[error("model refuted: {0}")]
    ModelRefuted(String),
    #[error("unauthorized execution: {0}")]
    Unauthorized(String),
    #[error("NATS request failed: {0}")]
    Nats(#[from] async_nats::Error),
    #[error("serialization failed: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// The policy boundary is the executable equivalent of
/// `can_execute(TargetTool, AgentRole)`.
///
/// Implementations must be deterministic and must not infer permissions from
/// model-generated text. A production deployment can back this trait with a
/// separately verified Prolog/Dafny service.
pub trait ExecutionBoundary: Send + Sync {
    fn can_execute(&self, tool: &ToolContract, role: &str, arguments: &Value) -> Result<(), String>;
}

pub struct AllowlistBoundary {
    allowed: BTreeMap<String, Vec<String>>,
}

impl AllowlistBoundary {
    pub fn new(allowed: BTreeMap<String, Vec<String>>) -> Self {
        Self { allowed }
    }
}

impl ExecutionBoundary for AllowlistBoundary {
    fn can_execute(&self, tool: &ToolContract, role: &str, _arguments: &Value) -> Result<(), String> {
        match self.allowed.get(&tool.name) {
            Some(roles) if roles.iter().any(|r| r == role) => Ok(()),
            _ => Err(format!("role `{role}` is not allowed to execute `{}`", tool.name)),
        }
    }
}

/// A bounded, explicit-input crystallizer. It does not accept a reasoning
/// prefix because hidden reasoning is not a valid MCP transport field.
pub trait DafnyCrystallizer: Send + Sync {
    fn crystallize(&self, invocation: &ToolInvocation) -> Result<(), String>;
}

pub struct StrictCrystallizer {
    pub max_argument_bytes: usize,
}

impl DafnyCrystallizer for StrictCrystallizer {
    fn crystallize(&self, invocation: &ToolInvocation) -> Result<(), String> {
        let bytes = serde_json::to_vec(&invocation.arguments)
            .map_err(|e| e.to_string())?;
        if bytes.len() > self.max_argument_bytes {
            return Err("argument entropy bound exceeded".into());
        }
        reject_reasoning_keys(&invocation.arguments)
    }
}

fn reject_reasoning_keys(value: &Value) -> Result<(), String> {
    match value {
        Value::Object(object) => {
            for (key, child) in object {
                let normalized = key.to_ascii_lowercase();
                if matches!(
                    normalized.as_str(),
                    "thinking" | "reasoning" | "chain_of_thought" | "cot" | "interleaved_thinking"
                ) {
                    return Err(format!("non-executable reasoning field `{key}` present"));
                }
                reject_reasoning_keys(child)?;
            }
        }
        Value::Array(items) => {
            for item in items {
                reject_reasoning_keys(item)?;
            }
        }
        _ => {}
    }
    Ok(())
}

pub struct NatsNode<B, C> {
    client: Arc<Mutex<Client>>,
    boundary: Arc<B>,
    crystallizer: Arc<C>,
    contracts: Arc<BTreeMap<String, ToolContract>>,
}

impl<B, C> NatsNode<B, C>
where
    B: ExecutionBoundary + 'static,
    C: DafnyCrystallizer + 'static,
{
    pub fn new(client: Client, contracts: ContractSet, boundary: B, crystallizer: C) -> Self {
        let contracts = contracts
            .tools
            .into_iter()
            .map(|tool| (tool.name.clone(), tool))
            .collect();
        Self {
            client: Arc::new(Mutex::new(client)),
            boundary: Arc::new(boundary),
            crystallizer: Arc::new(crystallizer),
            contracts: Arc::new(contracts),
        }
    }

    /// Extract contracts from an explicit `tools/list` JSON-RPC result.
    /// This performs metadata extraction only; it does not call the MCP server.
    pub fn extract_contracts(result: &Value) -> Result<ContractSet, AdapterError> {
        let tools = result
            .get("tools")
            .and_then(Value::as_array)
            .ok_or_else(|| AdapterError::InvalidEnvelope("missing result.tools[]".into()))?;
        let mut extracted = Vec::with_capacity(tools.len());
        for raw in tools {
            let name = raw
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| AdapterError::InvalidEnvelope("tool.name is required".into()))?;
            let input_schema = raw
                .get("inputSchema")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object"}));
            extracted.push(ToolContract {
                name: name.to_owned(),
                description: raw.get("description").and_then(Value::as_str).map(str::to_owned),
                input_schema,
            });
        }
        Ok(ContractSet { tools: extracted })
    }

    /// Validates and publishes a generate request. No MCP tool is executed here.
    pub async fn process(&self, invocation: ToolInvocation) -> Result<Message, AdapterError> {
        let contract = self
            .contracts
            .get(&invocation.tool)
            .ok_or_else(|| AdapterError::UnknownTool(invocation.tool.clone()))?
            .clone();

        self.crystallizer
            .crystallize(&invocation)
            .map_err(AdapterError::ModelRefuted)?;

        self.boundary
            .can_execute(&contract, &invocation.agent_role, &invocation.arguments)
            .map_err(AdapterError::Unauthorized)?;

        self.seal(&SealRequest {
            trace_id: invocation.trace_id.clone(),
            tool: invocation.tool.clone(),
            state: TraceState::Attested,
            metadata: BTreeMap::new(),
        })
        .await?;

        let request = GenerateRequest { contract, invocation };
        let payload = serde_json::to_vec(&request)?;
        let client = self.client.lock().await;
        client
            .request(GENERATE_SUBJECT, payload.into())
            .await
            .map_err(AdapterError::from)
    }

    pub async fn attest(&self, request: &AttestRequest) -> Result<Attestation, AdapterError> {
        let contract = self
            .contracts
            .get(&request.tool)
            .ok_or_else(|| AdapterError::UnknownTool(request.tool.clone()))?;
        let result = match self.crystallizer.crystallize(&ToolInvocation {
            tool: request.tool.clone(),
            arguments: request.arguments.clone(),
            agent_role: request.agent_role.clone(),
            trace_id: request.trace_id.clone(),
        }) {
            Ok(()) => self
                .boundary
                .can_execute(contract, &request.agent_role, &request.arguments)
                .map(|_| Attestation { allowed: true, reason: "policy accepted".into() })
                .unwrap_or_else(|reason| Attestation { allowed: false, reason }),
            Err(reason) => Attestation { allowed: false, reason },
        };
        let payload = serde_json::to_vec(request)?;
        let client = self.client.lock().await;
        let _ = client.request(ATTEST_SUBJECT, payload.into()).await?;
        Ok(result)
    }

    pub async fn seal(&self, request: &SealRequest) -> Result<(), AdapterError> {
        let payload = serde_json::to_vec(request)?;
        let client = self.client.lock().await;
        let _ = client.request(SEAL_SUBJECT, payload.into()).await?;
        Ok(())
    }

    /// Accept only an explicit JSON-RPC tools/call envelope. Any other method,
    /// including provider-specific hidden-thinking prefixes, is rejected.
    pub async fn handle_rpc(&self, envelope: RpcEnvelope) -> Result<Message, AdapterError> {
        if envelope.jsonrpc != "2.0" {
            return Err(AdapterError::InvalidEnvelope("jsonrpc must be 2.0".into()));
        }
        if envelope.method != "tools/call" {
            return Err(AdapterError::UnsupportedMethod(envelope.method));
        }
        let params = envelope
            .params
            .as_object()
            .ok_or_else(|| AdapterError::InvalidEnvelope("tools/call.params must be an object".into()))?;
        let tool = params
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| AdapterError::InvalidEnvelope("tools/call.params.name is required".into()))?;
        let arguments = params.get("arguments").cloned().unwrap_or_else(|| Value::Object(Map::new()));
        let agent_role = params
            .get("agentRole")
            .and_then(Value::as_str)
            .ok_or_else(|| AdapterError::InvalidEnvelope("agentRole is required".into()))?;
        let trace_id = params
            .get("traceId")
            .and_then(Value::as_str)
            .ok_or_else(|| AdapterError::InvalidEnvelope("traceId is required".into()))?;
        self.process(ToolInvocation {
            tool: tool.into(),
            arguments,
            agent_role: agent_role.into(),
            trace_id: trace_id.into(),
        })
        .await
    }
}
