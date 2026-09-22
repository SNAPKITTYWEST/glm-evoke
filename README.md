# GLM-Transformer-Evoke

Sovereign, local-first pipeline for ingesting GLM checkpoints, sealing weights WORM, and evoking an observable execution graph.

## Layout

- `src/config.rs` — `GlmConfig` parsing + invariant validation
- `src/worm.rs` — Write-Once-Read-Many memory manager (staging -> seal -> RO)
- `src/transform.rs` — parallel safetensors ingest, GLM key mapping, shape assertion before alloc, fused QKV + MLP gate/up splits
- `src/graph.rs` — sealed `ExecutionGraph` + per-layer views
- `src/evoke.rs` — `Evoker::evoke()` (ingest -> seal -> compile) and observable `generate_first_token()` loop
- `src/backend.rs` — `TensorBackend` trait (`CpuBackend` included; plug vLLM / custom engine)
- `src/telemetry.rs` — `TelemetrySink`, auditable `ExecutionRecord` stream

## Build

```bash
cargo build
cargo test
```

## Use

```rust
let cfg = GlmConfig::from_json_str(&std::fs::read_to_string("config.json")?)?;
let evoker = Evoker::evoke(cfg, vec!["model.safetensors".into()], Arc::new(CpuBackend), Arc::new(TracingSink::new())).await?;
let (tok, mut records) = evoker.generate_first_token(vec![1, 2, 3]).await?;
```

Every phase (`map`, `seal`, `compile`, `forward`, `sample`) emits an `ExecutionRecord` — no hidden chain-of-thought.

## Supplied companion modules

- `src/dual_mirror.rs`: NATS mirror and trace hashing module.
- `src/nats_node.rs`: MCP JSON-RPC to NATS adapter.
- `verify/quantum_agent.pl`: supplied Prolog policy and trace predicates.
- `verify/forge.dfy`: supplied Dafny specification.
- `docs/Checkpoint_Dynamics_Architecture.md`: supplied architecture document.
- `supplied/mercury-wire.Cargo.toml`: original downloaded Cargo manifest, retained separately from the active GLM crate manifest.
- `supplied/glm-evoke.zip`: unchanged source archive.

The companion Rust modules are exported by this crate. NATS operations require a running NATS server. The Prolog and Dafny files are separate artifacts and are not executed by Cargo.

Original companion Rust files are retained in supplied/. The active copies include NATS error-conversion compatibility fixes. The archive source has a borrow-checker fix in src/evoke.rs.

