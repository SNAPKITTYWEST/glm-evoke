use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::task::JoinSet;
use safetensors::SafeTensors;

use crate::config::GlmConfig;
use crate::error::EvokeError;
use crate::telemetry::{ExecutionRecord, TelemetrySink};
use crate::worm::{DType, TensorView, WormMemoryManager, SealedRegion};
use crate::graph::GlmBlockView;

fn canonical_glm_key(raw: &str) -> Option<(usize, &'static str)> {
    // Supports GLM-4 style: transformer.encoder.layers.{i}.self_attention.query_key_value.weight
    let layer_marker = ".layers.";
    let pos = raw.find(layer_marker)?;
    let rest = &raw[pos + layer_marker.len()..];
    let dot = rest.find('.')?;
    let layer_idx: usize = rest[..dot].parse().ok()?;

    let suffix: &'static str = if raw.contains("query_key_value.weight") {
        "attn.qkv"
    } else if raw.contains("self_attention.dense.weight") {
        "attn.o_proj"
    } else if raw.contains("dense_h_to_4h.weight") {
        "mlp.gate_up"
    } else if raw.contains("dense_4h_to_h.weight") {
        "mlp.down"
    } else if raw.contains("input_layernorm.weight") {
        "input_norm"
    } else if raw.contains("post_attention_layernorm.weight") {
        "post_norm"
    } else {
        return None;
    };
    Some((layer_idx, suffix))
}

fn assert_expected_shape(cfg: &GlmConfig, suffix: &str, got: &[usize], name: &str) -> Result<(), EvokeError> {
    let expected: Vec<usize> = match suffix {
        "attn.qkv" => vec![cfg.qkv_fused_dim(), cfg.hidden_size],
        "attn.o_proj" => vec![cfg.hidden_size, cfg.hidden_size],
        "mlp.gate_up" => vec![cfg.intermediate_size * 2, cfg.hidden_size],
        "mlp.down" => vec![cfg.hidden_size, cfg.intermediate_size],
        "input_norm" | "post_norm" => vec![cfg.hidden_size],
        _ => return Err(EvokeError::Config(format!("no shape rule for {suffix} ({name})"))),
    };
    if got != expected.as_slice() {
        return Err(EvokeError::ShapeMismatch { key: name.to_string(), expected, got: got.to_vec() });
    }
    Ok(())
}

fn dtype_of(view: &safetensors::tensor::TensorView) -> DType {
    use safetensors::Dtype as S;
    match view.dtype() {
        S::F32 => DType::F32,
        S::F16 => DType::F16,
        S::BF16 => DType::BF16,
        _ => DType::BF16,
    }
}

fn split_fused_qkv(data: &[u8], shape: &[usize], cfg: &GlmConfig) -> Result<Vec<(&'static str, Vec<u8>, Vec<usize>)>, EvokeError> {
    // shape: [qkv_fused_dim, hidden_size], row-major bytes (assume 2 bytes/elem for F16/BF16, 4 for F32)
    // We split rows into q / k / v. Byte-exact, no reinterpretation here — backend dequantizes.
    let rows = shape[0];
    let cols = shape[1];
    let elem_size = data.len() / (rows * cols);
    if elem_size != 2 && elem_size != 4 {
        return Err(EvokeError::Config(format!("unexpected elem size {elem_size} for qkv")));
    }
    let q_rows = cfg.hidden_size;
    let kv_rows = cfg.kv_dim();
    if rows != q_rows + 2 * kv_rows {
        return Err(EvokeError::Config(format!("qkv row mismatch: {rows} vs {}", q_rows + 2 * kv_rows)));
    }
    let row_bytes = cols * elem_size;
    let q_end = q_rows * row_bytes;
    let k_end = q_end + kv_rows * row_bytes;

    let q = data[..q_end].to_vec();
    let k = data[q_end..k_end].to_vec();
    let v = data[k_end..].to_vec();

    Ok(vec![
        ("q", q, vec![q_rows, cols]),
        ("k", k, vec![kv_rows, cols]),
        ("v", v, vec![kv_rows, cols]),
    ])
}

fn split_mlp_gate_up(data: Vec<u8>, shape: &[usize], cfg: &GlmConfig) -> Result<Vec<(&'static str, Vec<u8>, Vec<usize>)>, EvokeError> {
    // [2*intermediate, hidden] -> gate, up each [intermediate, hidden]
    let rows = shape[0];
    let cols = shape[1];
    let elem_size = data.len() / (rows * cols);
    let half_rows = cfg.intermediate_size;
    if rows != half_rows * 2 {
        return Err(EvokeError::Config(format!("mlp gate_up row mismatch {rows}")));
    }
    let half_bytes = half_rows * cols * elem_size;
    let up = data[half_bytes..].to_vec();
    let gate = data[..half_bytes].to_vec();
    Ok(vec![
        ("gate", gate, vec![half_rows, cols]),
        ("up", up, vec![half_rows, cols]),
    ])
}

pub async fn ingest_checkpoint(
    shards: Vec<PathBuf>,
    config: Arc<GlmConfig>,
    telemetry: Arc<dyn TelemetrySink>,
) -> Result<(Arc<SealedRegion>, Vec<GlmBlockView>), EvokeError> {
    config.validate()?;
    let total_elems: usize = config.num_layers * (config.qkv_fused_dim() * config.hidden_size + config.hidden_size * config.hidden_size + 2 * config.intermediate_size * config.hidden_size);
    let mut mgr = WormMemoryManager::new(total_elems * 4 + 1024);

    let mut set = JoinSet::new();
    for shard in shards {
        let cfg = config.clone();
        let tel = telemetry.clone();
        set.spawn(async move {
            let bytes = tokio::fs::read(&shard).await
                .map_err(|e| EvokeError::Shard { shard: shard.display().to_string(), msg: e.to_string() })?;
            let st = SafeTensors::deserialize(&bytes)
                .map_err(|e| EvokeError::Shard { shard: shard.display().to_string(), msg: e.to_string() })?;
            let mut out: Vec<(String, Vec<u8>, Vec<usize>, DType)> = Vec::new();
            for (name, view) in st.tensors() {
                let (layer_idx, suffix) = canonical_glm_key(&name)
                    .ok_or_else(|| EvokeError::Config(format!("unknown key: {name}")))?;
                if layer_idx >= cfg.num_layers {
                    return Err(EvokeError::ShapeMismatch {
                        key: name.clone(), expected: vec![cfg.num_layers], got: vec![layer_idx],
                    });
                }
                assert_expected_shape(&cfg, suffix, view.shape(), &name)?;
                let dt = dtype_of(&view);
                if suffix == "attn.qkv" {
                    for (tag, data, shape) in split_fused_qkv(view.data(), view.shape(), &cfg)? {
                        out.push((format!("layer.{layer_idx}.attn.{tag}"), data, shape, dt));
                    }
                } else if suffix == "mlp.gate_up" {
                    for (tag, data, shape) in split_mlp_gate_up(view.data().to_vec(), view.shape(), &cfg)? {
                        out.push((format!("layer.{layer_idx}.mlp.{tag}"), data, shape, dt));
                    }
                } else {
                    let key = match suffix {
                        "attn.o_proj" => format!("layer.{layer_idx}.attn.o_proj"),
                        "mlp.down" => format!("layer.{layer_idx}.mlp.down"),
                        "input_norm" => format!("layer.{layer_idx}.input_norm"),
                        "post_norm" => format!("layer.{layer_idx}.post_norm"),
                        _ => unreachable!(),
                    };
                    out.push((key, view.data().to_vec(), view.shape().to_vec(), dt));
                }
                tel.emit(ExecutionRecord { seq: 0, phase: "map", layer: Some(layer_idx), detail: format!("mapped {name}") });
            }
            Ok::<_, EvokeError>(out)
        });
    }

    let mut pending: HashMap<String, (Vec<u8>, Vec<usize>, DType)> = HashMap::new();
    while let Some(res) = set.join_next().await {
        let items = res.map_err(|e| EvokeError::Backend(e.to_string()))??;
        for (k, d, s, dt) in items {
            if pending.insert(k.clone(), (d, s, dt)).is_some() {
                return Err(EvokeError::Config(format!("duplicate tensor key: {k}")));
            }
        }
    }

    let mut layers = Vec::with_capacity(config.num_layers);
    for idx in 0..config.num_layers {
        let mut take = |s: &str| -> Result<TensorView, EvokeError> {
            let key = format!("layer.{idx}.{s}");
            let (data, shape, dtype) = pending.remove(&key)
                .ok_or_else(|| EvokeError::Config(format!("missing {key}")))?;
            mgr.alloc(&data, &key, shape, dtype)
        };
        layers.push(GlmBlockView {
            layer_idx: idx,
            q: take("attn.q")?,
            k: take("attn.k")?,
            v: take("attn.v")?,
            o_proj: take("attn.o_proj")?,
            mlp_gate: take("mlp.gate")?,
            mlp_up: take("mlp.up")?,
            mlp_down: take("mlp.down")?,
            input_norm: take("input_norm")?,
            post_norm: take("post_norm")?,
        });
    }

    telemetry.emit(ExecutionRecord { seq: 0, phase: "seal", layer: None, detail: "sealing WORM region".into() });
    let sealed = mgr.seal()?;
    Ok((sealed, layers))
}
