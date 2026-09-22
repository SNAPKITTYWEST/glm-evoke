use thiserror::Error;

#[derive(Debug, Error)]
pub enum EvokeError {
    #[error("config error: {0}")]
    Config(String),
    #[error("shape mismatch for {key}: expected {expected:?}, got {got:?}")]
    ShapeMismatch { key: String, expected: Vec<usize>, got: Vec<usize> },
    #[error("shard error {shard}: {msg}")]
    Shard { shard: String, msg: String },
    #[error("seal violation: {0}")]
    Seal(String),
    #[error("backend error: {0}")]
    Backend(String),
    #[error("io error: {0}")]
    Io(String),
}
