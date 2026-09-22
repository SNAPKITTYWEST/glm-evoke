module Forge {
  const EntropyBound: real := 0.20

  datatype TraceStatus = Running | Refuted | Committed
  type Vector = seq<real>
  type Matrix = seq<Vector>

  class TraceRecord {
    var seqId: string
    var depth: nat
    var state: Vector
    var operator: Matrix
    var entropy: real
    var action: string
    var prevHash: string
    var hash: string

    constructor (sid: string, d: nat, s: Vector, q: Matrix, e: real,
                 a: string, previous: string, current: string)
      ensures seqId == sid && depth == d && state == s && operator == q
      ensures entropy == e && action == a && prevHash == previous && hash == current
    {
      seqId, depth, state, operator, entropy, action, prevHash, hash :=
        sid, d, s, q, e, a, previous, current;
    }
  }

  predicate Square(q: Matrix)
    reads *
  { |q| > 0 && forall i :: 0 <= i < |q| ==> |q[i]| == |q| }

  predicate FiniteVector(v: Vector)
  { forall i :: 0 <= i < |v| ==> v[i] == v[i] }

  predicate FiniteMatrix(q: Matrix)
    reads *
  { Square(q) && forall i, j :: 0 <= i < |q| && 0 <= j < |q| ==> q[i][j] == q[i][j] }

  predicate DimensionsStable(previous: Vector, next: Vector, q: Matrix)
    reads *
  { |previous| > 0 && |next| == |previous| && |q| == |previous| && Square(q) }

  predicate Invariant(previous: Vector, next: Vector, q: Matrix, entropy: real)
    reads *
  { FiniteVector(previous) && FiniteVector(next) && FiniteMatrix(q) &&
    DimensionsStable(previous, next, q) && entropy == entropy && entropy <= EntropyBound }

  predicate HashChainStep(previous: TraceRecord, current: TraceRecord)
    reads previous, current
  { current.seqId == previous.seqId && current.depth == previous.depth + 1 &&
    current.prevHash == previous.hash }

  predicate TracePrefixValid(prefix: seq<TraceRecord>)
    reads *
  { |prefix| > 0 && forall i :: 0 < i < |prefix| ==> HashChainStep(prefix[i-1], prefix[i]) }

  predicate CommitAllowed(previous: TraceRecord, current: TraceRecord)
    reads previous, current
  { HashChainStep(previous, current) && current.entropy <= EntropyBound &&
    FiniteVector(current.state) && FiniteMatrix(current.operator) }

  method VerifyOrRefute(previous: TraceRecord, current: TraceRecord)
      returns (status: TraceStatus)
    requires previous != null && current != null
    ensures status == Committed <==> CommitAllowed(previous, current)
    ensures status == Refuted <==> !CommitAllowed(previous, current)
  {
    if CommitAllowed(previous, current) { return Committed; }
    return Refuted;
  }
}
