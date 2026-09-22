use async_nats::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fmt, sync::Arc};
use tokio::sync::Mutex;

pub const GENERATE_SUBJECT: &str = "glm.evoke.generate";
pub const ATTEST_SUBJECT: &str = "glm.evoke.attest";
pub const SEAL_SUBJECT: &str = "glm.evoke.seal";
pub const ENTROPY_BOUND: f64 = 0.20;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TraceRecord {
    pub seq_id: String,
    pub depth: u64,
    pub state: Vec<f64>,
    pub operator: Vec<Vec<f64>>,
    pub entropy: f64,
    pub prev_hash: String,
    pub hash: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecTrace {
    pub seq_id: String,
    pub records: Vec<TraceRecord>,
    pub head: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectModule {
    pub event: &'static str,
    pub seq_id: String,
    pub depth: u64,
    pub delta_hash: String,
    pub alpha: TraceRecord,
    pub beta: TraceRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Committed,
    Refuted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MirrorResult {
    pub status: Status,
    pub alpha: ExecTrace,
    pub beta: ExecTrace,
}

#[derive(Debug)]
pub enum MirrorError {
    Execution(String),
    Serialization(serde_json::Error),
    Nats(async_nats::Error),
    InvalidTensor(String),
    Refuted(RejectModule),
}

impl fmt::Display for MirrorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Execution(e) => write!(f, "execution failed: {e}"),
            Self::Serialization(e) => write!(f, "serialization failed: {e}"),
            Self::Nats(e) => write!(f, "NATS failed: {e}"),
            Self::InvalidTensor(e) => write!(f, "invalid tensor: {e}"),
            Self::Refuted(e) => write!(f, "mirror refuted at {}:{}", e.seq_id, e.depth),
        }
    }
}

impl std::error::Error for MirrorError {}
impl From<serde_json::Error> for MirrorError {
    fn from(value: serde_json::Error) -> Self { Self::Serialization(value) }
}
impl From<async_nats::Error> for MirrorError {
    fn from(value: async_nats::Error) -> Self { Self::Nats(value) }
}

/// The primary and mirror receive the same immutable input. Neither is allowed
/// to mutate the live WORM arena; commit happens only after the convergence gate.
#[async_trait::async_trait]
pub trait MirrorExecutor: Send + Sync + 'static {
    async fn execute(
        &self,
        seq_id: &str,
        input: Arc<Vec<f64>>,
        previous_hash: &str,
    ) -> Result<ExecTrace, MirrorError>;
}

#[async_trait::async_trait]
pub trait WormArena: Send + Sync {
    async fn commit(&self, trace: &ExecTrace) -> Result<(), MirrorError>;
    async fn quarantine(&self, seq_id: &str, depth: u64) -> Result<(), MirrorError>;
    async fn drop_context(&self, seq_id: &str) -> Result<(), MirrorError>;
}

pub struct MirrorDispatcher<P, M, W> {
    pub nats: Client,
    pub primary: Arc<P>,
    pub mirror: Arc<M>,
    pub worm: Arc<W>,
}

impl<P, M, W> MirrorDispatcher<P, M, W>
where
    P: MirrorExecutor,
    M: MirrorExecutor,
    W: WormArena,
{
    pub async fn dispatch(
        &self,
        seq_id: impl Into<String>,
        input: Vec<f64>,
        previous_hash: impl Into<String>,
    ) -> Result<MirrorResult, MirrorError> {
        let seq_id = seq_id.into();
        let previous_hash = previous_hash.into();
        let frozen_input = Arc::new(input);

        // Both branches start before either result is observed.
        let primary = self.primary.execute(
            &seq_id,
            Arc::clone(&frozen_input),
            &previous_hash,
        );
        let mirror = self.mirror.execute(
            &seq_id,
            Arc::clone(&frozen_input),
            &previous_hash,
        );
        let (alpha, beta) = tokio::join!(primary, mirror);
        let alpha = alpha?;
        let beta = beta?;

        let result = self.convergence_gate(&alpha, &beta).await;
        match result {
            Ok(()) => {
                // The only live-arena mutation is after full trace validation.
                self.worm.commit(&alpha).await?;
                self.publish_trace(GENERATE_SUBJECT, &alpha).await?;
                self.publish_trace(ATTEST_SUBJECT, &beta).await?;
                self.publish_seal(&seq_id, Status::Committed).await?;
                Ok(MirrorResult { status: Status::Committed, alpha, beta })
            }
            Err(reject) => {
                self.worm.quarantine(&seq_id, reject.depth).await?;
                self.worm.drop_context(&seq_id).await?;
                self.nats
                    .publish(SEAL_SUBJECT, serde_json::to_vec(&reject)?.into())
                    .await?;
                Err(MirrorError::Refuted(reject))
            }
        }
    }

    async fn convergence_gate(
        &self,
        alpha: &ExecTrace,
        beta: &ExecTrace,
    ) -> Result<(), RejectModule> {
        if alpha.records.len() != beta.records.len() {
            return Err(self.reject_for_length_mismatch(alpha, beta));
        }
        for (a, b) in alpha.records.iter().zip(&beta.records) {
            let entropy = symmetrized_entropy(&a.operator)
                .map_err(|_| self.reject_at(a, b, "invalid operator"))?;
            if entropy > ENTROPY_BOUND {
                return Err(self.reject_at(a, b, "entropy boundary exceeded"));
            }
            if a.hash != b.hash {
                return Err(self.reject_at(a, b, "mirror hash mismatch"));
            }
            if a.prev_hash != b.prev_hash || a.state != b.state || a.depth != b.depth {
                return Err(self.reject_at(a, b, "state record mismatch"));
            }
        }
        if alpha.head != beta.head {
            return Err(self.reject_at(
                alpha.records.last().unwrap_or(&empty_record()),
                beta.records.last().unwrap_or(&empty_record()),
                "trace head mismatch",
            ));
        }
        Ok(())
    }

    async fn publish_trace(&self, subject: &str, trace: &ExecTrace) -> Result<(), MirrorError> {
        self.nats.publish(subject.to_owned(), serde_json::to_vec(trace)?.into()).await?;
        Ok(())
    }

    async fn publish_seal(&self, seq_id: &str, status: Status) -> Result<(), MirrorError> {
        #[derive(Serialize)]
        struct Seal<'a> { seq_id: &'a str, status: Status }
        self.nats.publish(SEAL_SUBJECT, serde_json::to_vec(&Seal { seq_id, status })?.into()).await?;
        Ok(())
    }

    fn reject_at(&self, a: &TraceRecord, b: &TraceRecord, reason: &str) -> RejectModule {
        RejectModule {
            event: "RejectModule",
            seq_id: a.seq_id.clone(),
            depth: a.depth,
            delta_hash: sha256_hex(format!("{}:{}:{}", a.hash, b.hash, reason).as_bytes()),
            alpha: a.clone(),
            beta: b.clone(),
        }
    }

    fn reject_for_length_mismatch(&self, a: &ExecTrace, b: &ExecTrace) -> RejectModule {
        let ar = a.records.last().cloned().unwrap_or_else(empty_record);
        let br = b.records.last().cloned().unwrap_or_else(empty_record);
        self.reject_at(&ar, &br, "trace length mismatch")
    }
}

fn empty_record() -> TraceRecord {
    TraceRecord { seq_id: String::new(), depth: 0, state: vec![], operator: vec![], entropy: 0.0, prev_hash: String::new(), hash: String::new() }
}

/// Q = (Q + Q^T)/2, followed by Shannon entropy over normalized absolute entries.
pub fn symmetrized_entropy(q: &[Vec<f64>]) -> Result<f64, MirrorError> {
    let n = q.len();
    if n == 0 || q.iter().any(|row| row.len() != n) {
        return Err(MirrorError::InvalidTensor("operator must be non-empty and square".into()));
    }
    let mut values = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            values.push(((q[i][j] + q[j][i]) / 2.0).abs());
        }
    }
    let total: f64 = values.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Ok(0.0);
    }
    Ok(values.into_iter().filter(|v| *v > 0.0).map(|v| {
        let p = v / total;
        -p * p.ln()
    }).sum())
}

pub fn hash_record(
    seq_id: &str,
    depth: u64,
    state: &[f64],
    operator: &[Vec<f64>],
    entropy: f64,
    prev_hash: &str,
) -> Result<String, MirrorError> {
    #[derive(Serialize)]
    struct Canonical<'a> {
        seq_id: &'a str,
        depth: u64,
        state: &'a [f64],
        operator: &'a [Vec<f64>],
        entropy: f64,
        prev_hash: &'a str,
    }
    let bytes = serde_json::to_vec(&Canonical { seq_id, depth, state, operator, entropy, prev_hash })?;
    Ok(sha256_hex(&bytes))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(bytes);
    hex::encode(h.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_uses_symmetric_operator() {
        let q = vec![vec![1.0, 0.0], vec![2.0, 0.0]];
        let h = symmetrized_entropy(&q).unwrap();
        assert!(h >= 0.0 && h.is_finite());
    }

    #[test]
    fn hash_chain_changes_when_parent_changes() {
        let a = hash_record("s", 0, &[1.0], &[vec![1.0]], 0.0, "0").unwrap();
        let b = hash_record("s", 1, &[1.0], &[vec![1.0]], 0.0, &a).unwrap();
        let c = hash_record("s", 1, &[1.0], &[vec![1.0]], 0.0, "different").unwrap();
        assert_ne!(b, c);
    }
}

// Optional adapter state for callers that want a mutex-protected local status.
pub type SharedStatus = Arc<Mutex<Status>>;

impl From<async_nats::PublishError> for MirrorError {
    fn from(value: async_nats::PublishError) -> Self { Self::Nats(Box::new(value)) }
}
