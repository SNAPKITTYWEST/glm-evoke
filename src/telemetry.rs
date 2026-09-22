use std::sync::atomic::{AtomicU64, Ordering};

#[derive(Debug, Clone)]
pub struct ExecutionRecord {
    pub seq: u64,
    pub phase: &'static str,
    pub layer: Option<usize>,
    pub detail: String,
}

pub trait TelemetrySink: Send + Sync {
    fn emit(&self, record: ExecutionRecord);
}

pub struct TracingSink {
    counter: AtomicU64,
}

impl TracingSink {
    pub fn new() -> Self {
        Self { counter: AtomicU64::new(1) }
    }
}

impl TelemetrySink for TracingSink {
    fn emit(&self, mut record: ExecutionRecord) {
        let seq = self.counter.fetch_add(1, Ordering::SeqCst);
        record.seq = seq;
        tracing::info!(seq, phase = record.phase, layer = ?record.layer, detail = %record.detail, "evoke");
    }
}

pub struct ChannelSink {
    pub tx: tokio::sync::mpsc::Sender<ExecutionRecord>,
    counter: AtomicU64,
}

impl ChannelSink {
    pub fn new(tx: tokio::sync::mpsc::Sender<ExecutionRecord>) -> Self {
        Self { tx, counter: AtomicU64::new(1) }
    }
}

impl TelemetrySink for ChannelSink {
    fn emit(&self, mut record: ExecutionRecord) {
        record.seq = self.counter.fetch_add(1, Ordering::SeqCst);
        let _ = self.tx.try_send(record);
    }
}
