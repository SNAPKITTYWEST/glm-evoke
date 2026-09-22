# Checkpoint Dynamics Architecture

## Formal object

The execution model is the tuple **M = (X, O, R, D, Φ)**. The state space `X` contains the finite activation/state vector, the current cycle, the phase delay, and the latest audit hash. Operators `O` are policy-checked actions, symmetric numerical operators, routing transforms, and append operations. Relations `R` are the ordered dependency edges between transitions and their predecessor hash. Domain bounds `D` include finite numeric values, stable dimensions, `H(Q) ≤ 0.20`, checked arena capacity, and checked cycle arithmetic. The transition map `Φ` computes `next = Q · state + route(embedding, phase_delay)`.

At step `t`, the loop computes `x[t+1] = Φ(x[t], o[t])` and emits an observable `y[t]` as either `Committed{proof}` or `Refuted{fault}`. A commit is permitted only after the invariant predicate `I` succeeds. Rejected transitions do not change the state, cycle, hash, or arena usage.

## Invariants

The implementation enforces the following invariant before committing a transition:

`I(previous, next, Q, H) := finite(previous) ∧ finite(next) ∧ finite(Q) ∧ dimensions_stable ∧ H(Q) ≤ 0.20 ∧ permitted(action) ∧ append_only ∧ hash_chain`.

The Rust implementation checks all computationally decidable portions. The Dafny specification states the corresponding proof obligations over mathematical reals and sequences. Floating-point finiteness is checked at runtime; the Dafny `x == x` idiom models the absence of NaN in the executable boundary contract.

## Storage and audit chain

`WormArena` is append-only and bounded. It exposes `append` and committed-range `read`, but no overwrite, seek-write, truncate, or clear operation. Capacity exhaustion is a fault. Each serialized record contains the cycle, next state, normalized operator, entropy, action, predecessor hash, and current hash. The current hash is SHA-256 over those fields and the predecessor hash, yielding a tamper-evident chain.

This is **logical WORM behavior**, not a claim about immutable hardware or a privileged operating-system memory region. `mlock` is optional and only requests page residency; failure is reported rather than hidden.

## Fault semantics

Policy denial returns an error before numerical execution. An invariant failure returns `Refuted` and performs no commit. Serialization, capacity, arithmetic overflow, allocation, and memory-lock failures are hard errors. No refutation path clears or mutates committed records.

## Verification

Run the tests with:

```text
cargo test --manifest-path /home/ubuntu/output/Cargo.toml
```

The tests cover append-only capacity, policy non-execution, valid hash-chain commit, and entropy refutation without state or arena mutation.
