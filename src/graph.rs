use std::sync::Arc;
use crate::config::GlmConfig;
use crate::telemetry::TelemetrySink;
use crate::worm::{SealedRegion, TensorView};

#[derive(Debug, Clone)]
pub struct GlmBlockView {
    pub layer_idx: usize,
    pub q: TensorView,
    pub k: TensorView,
    pub v: TensorView,
    pub o_proj: TensorView,
    pub mlp_gate: TensorView,
    pub mlp_up: TensorView,
    pub mlp_down: TensorView,
    pub input_norm: TensorView,
    pub post_norm: TensorView,
}

pub struct ExecutionGraph {
    pub config: Arc<GlmConfig>,
    pub weights: Arc<SealedRegion>,
    pub layers: Vec<GlmBlockView>,
    pub telemetry: Arc<dyn TelemetrySink>,
}

impl ExecutionGraph {
    pub fn new(config: Arc<GlmConfig>, weights: Arc<SealedRegion>, layers: Vec<GlmBlockView>, telemetry: Arc<dyn TelemetrySink>) -> Self {
        Self { config, weights, layers, telemetry }
    }
}
