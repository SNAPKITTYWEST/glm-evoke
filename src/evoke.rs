use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;

use crate::backend::TensorBackend;
use crate::config::GlmConfig;
use crate::error::EvokeError;
use crate::graph::ExecutionGraph;
use crate::telemetry::{ExecutionRecord, TelemetrySink};
use crate::transform::ingest_checkpoint;
use crate::worm::TensorView;

pub struct Evoker<B: TensorBackend> {
    pub graph: Arc<ExecutionGraph>,
    pub backend: Arc<B>,
}

impl<B: TensorBackend> Evoker<B> {
    pub async fn evoke(
        config: GlmConfig,
        shards: Vec<PathBuf>,
        backend: Arc<B>,
        telemetry: Arc<dyn TelemetrySink>,
    ) -> Result<Self, EvokeError> {
        let config = Arc::new(config);
        let (weights, layers) = ingest_checkpoint(shards, config.clone(), telemetry.clone()).await?;
        telemetry.emit(ExecutionRecord {
            seq: 0, phase: "compile", layer: None,
            detail: format!("graph compiled: {} layers, hidden {}", layers.len(), config.hidden_size),
        });
        let graph = Arc::new(ExecutionGraph::new(config, weights, layers, telemetry));
        Ok(Self { graph, backend })
    }

    fn dequant_f32(&self, v: &TensorView) -> Vec<f32> {
        // Production: zero-copy BF16/F16 -> F32 via backend kernel.
        // Scaffold: return zeros with correct count to keep shapes honest.
        let n: usize = v.shape.iter().product();
        vec![0.0; n]
    }

    pub async fn generate_first_token(
        &self,
        input_ids: Vec<u32>,
    ) -> Result<(u32, mpsc::Receiver<ExecutionRecord>), EvokeError> {
        let (tx, rx) = mpsc::channel(1024);
        let sink = Arc::new(crate::telemetry::ChannelSink::new(tx.clone()));
        // Bridge graph telemetry into the per-request channel as well
        let graph = self.graph.clone();
        let backend = self.backend.clone();

        let mut hidden = vec![0.0f32; graph.config.hidden_size];
        // toy embedding: hash ids into hidden
        for (i, id) in input_ids.iter().enumerate() {
            let index = i % hidden.len();
            hidden[index] += (*id as f32) * 0.001;
        }

        for layer in &graph.layers {
            let normed = backend.rms_norm(&hidden, &self.dequant_f32(&layer.input_norm), graph.config.rms_norm_eps);

            let q = backend.matmul(&normed, &self.dequant_f32(&layer.q), 1, graph.config.hidden_size, graph.config.hidden_size);
            let k = backend.matmul(&normed, &self.dequant_f32(&layer.k), 1, graph.config.hidden_size, graph.config.kv_dim());
            let v = backend.matmul(&normed, &self.dequant_f32(&layer.v), 1, graph.config.hidden_size, graph.config.kv_dim());

            // Simplified single-head attention placeholder with RoPE hook
            let mut qm = q.clone();
            backend.rope(&mut qm, 0, graph.config.head_dim(), graph.config.rope_theta);
            let attn_out = qm; // production: full MHA + softmax + o_proj
            let proj = backend.matmul(&attn_out, &self.dequant_f32(&layer.o_proj), 1, graph.config.hidden_size, graph.config.hidden_size);
            hidden = add(&hidden, &proj);

            let normed2 = backend.rms_norm(&hidden, &self.dequant_f32(&layer.post_norm), graph.config.rms_norm_eps);
            let gate = backend.matmul(&normed2, &self.dequant_f32(&layer.mlp_gate), 1, graph.config.hidden_size, graph.config.intermediate_size);
            let up = backend.matmul(&normed2, &self.dequant_f32(&layer.mlp_up), 1, graph.config.hidden_size, graph.config.intermediate_size);
            let gated: Vec<f32> = silu(&gate).iter().zip(up.iter()).map(|(a, b)| a * b).collect();
            let mlp = backend.matmul(&gated, &self.dequant_f32(&layer.mlp_down), 1, graph.config.intermediate_size, graph.config.hidden_size);
            hidden = add(&hidden, &mlp);

            let rec = ExecutionRecord { seq: 0, phase: "forward", layer: Some(layer.layer_idx), detail: format!("layer {} done", layer.layer_idx) };
            graph.telemetry.emit(rec.clone());
            sink.emit(rec);
            let _ = tx.send(ExecutionRecord { seq: 0, phase: "forward", layer: Some(layer.layer_idx), detail: "stream".into() }).await;
        }

        let token = (hidden[0] * 1000.0) as u32 % graph.config.vocab_size as u32;
        let rec = ExecutionRecord { seq: 0, phase: "sample", layer: None, detail: format!("first token {token}") };
        graph.telemetry.emit(rec.clone());
        Ok((token, rx))
    }
}

fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v / (1.0 + (-v).exp())).collect()
}
