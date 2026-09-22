use serde::Deserialize;
use crate::error::EvokeError;

#[derive(Debug, Clone, Deserialize)]
pub struct GlmConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_layers: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tie_word_embeddings: bool,
}

impl GlmConfig {
    pub fn from_json_str(s: &str) -> Result<Self, EvokeError> {
        serde_json::from_str(s).map_err(|e| EvokeError::Config(e.to_string()))
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    pub fn kv_dim(&self) -> usize {
        self.head_dim() * self.num_key_value_heads
    }

    pub fn qkv_fused_dim(&self) -> usize {
        self.hidden_size + 2 * self.kv_dim()
    }

    pub fn validate(&self) -> Result<(), EvokeError> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(EvokeError::Config("hidden_size must divide num_attention_heads".into()));
        }
        if self.num_layers == 0 {
            return Err(EvokeError::Config("num_layers must be > 0".into()));
        }
        Ok(())
    }
}
