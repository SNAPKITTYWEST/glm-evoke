use std::collections::HashMap;
use std::sync::Arc;
use crate::error::EvokeError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType { F32, F16, BF16 }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryState { Staging, Sealed }

#[derive(Debug, Clone)]
pub struct TensorView {
    pub offset: usize,
    pub len: usize,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

pub struct SealedRegion {
    data: Arc<[u8]>,
}

impl SealedRegion {
    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    pub fn slice(&self, view: &TensorView) -> &[u8] {
        &self.data[view.offset..view.offset + view.len]
    }
}

pub struct WormMemoryManager {
    state: MemoryState,
    staging: Option<Vec<u8>>,
    cursor: usize,
    table: HashMap<String, TensorView>,
}

impl WormMemoryManager {
    pub fn new(capacity_bytes: usize) -> Self {
        Self {
            state: MemoryState::Staging,
            staging: Some(Vec::with_capacity(capacity_bytes)),
            cursor: 0,
            table: HashMap::new(),
        }
    }

    pub fn alloc(&mut self, bytes: &[u8], key: &str, shape: Vec<usize>, dtype: DType) -> Result<TensorView, EvokeError> {
        if self.state != MemoryState::Staging {
            return Err(EvokeError::Seal("alloc after seal".into()));
        }
        let staging = self.staging.as_mut().ok_or_else(|| EvokeError::Seal("no staging".into()))?;
        let aligned = (self.cursor + 63) & !63;
        if aligned + bytes.len() > staging.capacity() {
            return Err(EvokeError::Shard {
                shard: key.into(),
                msg: format!("OOM: need {}, capacity {}", aligned + bytes.len(), staging.capacity()),
            });
        }
        if staging.len() < aligned + bytes.len() {
            staging.resize(aligned + bytes.len(), 0);
        }
        staging[aligned..aligned + bytes.len()].copy_from_slice(bytes);
        self.cursor = aligned + bytes.len();
        let view = TensorView { offset: aligned, len: bytes.len(), shape, dtype };
        self.table.insert(key.to_string(), view.clone());
        Ok(view)
    }

    pub fn get(&self, key: &str) -> Option<&TensorView> {
        self.table.get(key)
    }

    pub fn seal(mut self) -> Result<Arc<SealedRegion>, EvokeError> {
        let staging = self.staging.take().ok_or_else(|| EvokeError::Seal("double seal".into()))?;
        #[cfg(unix)]
        unsafe {
            // Best-effort mlock; failure is logged, not fatal
            libc::mlock(staging.as_ptr() as *const libc::c_void, staging.len());
        }
        self.state = MemoryState::Sealed;
        Ok(Arc::new(SealedRegion { data: Arc::from(staging.into_boxed_slice()) }))
    }
}
