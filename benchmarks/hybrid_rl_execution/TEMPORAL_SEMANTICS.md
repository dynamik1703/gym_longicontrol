# Temporal-Semantics Amendment — Hybrid RL V1

Status: **frozen pre-training amendment**. This revision corrects a mathematical
inconsistency in the design at source commit
`337fe498083eb3d4c63263f696ee8b30f6800fec`; it was fixed before main training
or any reserved Diagnostic/Validation policy evaluation.

## Defect in the prior description

The physical system advances at `dt = 0.1 s`, while the strategy selects one
target speed every ten native steps and holds it for one second. A replay entry
for every native step with a one-step SAC target would falsely make a new
strategy action available after `0.1 s`. That process is not the controller that
will be executed.

The executable study is therefore a decision-boundary SMDP. The shield still
runs independently at 10 Hz and retains final action authority.

## Discount interpretation

The frozen strategy discount is interpreted as

```text
gamma_decision = 0.99 per 1.0-second strategy decision
delta_native = gamma_decision^(1/10)
             = 0.9989954712917501 per 0.1-second native step
```

Thus `delta_native^10 = 0.99`. This gives an e-folding physical horizon of
approximately `99.5 s` and a half-life of approximately `69.0 s`. Interpreting
`0.99` at 10 Hz would instead give an e-folding horizon near `9.95 s`; that
interpretation is rejected because it conflicts with the frozen 1 Hz strategy.

Gamma was selected by temporal meaning, not by policy-performance results.

## Macro transition

At decision boundary `n`, the actor observes the frozen causal 16-vector
`S_n`, proposes normalized target `A_n ∈ [-1,1]`, and holds it for `k` native
steps, where `1 <= k <= 10`. Normally `k=10`; a true episode termination,
environment truncation, or the exact study-budget boundary can shorten it.

For native signal `z_i`, the stored block value is

```text
Z_n = sum(i=0..k-1) delta_native^i z_i.
```

For a nonterminal full block, the continuation discount is `0.99`. For a
nonterminal block of physical duration `0.1k s`, it is `delta_native^k`. A true
episode end or truncation has continuation discount zero. The exact final
study-budget cut is treated as a dataset-horizon truncation: it has no
bootstrap and no terminal-infeasibility charge.

The action in replay is always `A_n`, the proposed target-speed strategy
action. Executed accelerator/brake actions and shield decisions are transition
effects and diagnostics, never substituted for the strategy action.

## Boundaries and reset

- The actor samples exactly once at the start of each block.
- The same target is passed to the native action generator for every step in
  the block.
- The shield is evaluated anew on every native observation.
- Replay insertion occurs only after ten steps or a shortened terminal/block
  boundary.
- Episode reset starts a new block and cannot leak a held target.
- A checkpoint may occur inside a block. It retains the held target, remaining
  steps, partial discounted accumulators, current macro observation, controller
  memory, environment, stream cursor, learner/replay, and all RNG states.

## Update accounting amendment

Native interactions remain the scientific budget. Update credit accrues at the
frozen rate `0.1` per real native transition. No credit is redeemed before the
10,000-native-transition replay warm-up. At completed macro boundaries, the
cumulative target is

```text
U(N) = 0                         if N < 10,000
U(N) = min(30,000, floor(N/10))  otherwise.
```

Consequently the deferred warm-up credit is deterministically redeemed at the
first eligible macro boundary. No environment step or imagined transition is
added, and `U(300,000)=30,000` exactly. This resolves the otherwise inconsistent
simultaneous requirements of a 10k warm-up, `0.1` update/native-step credit, and
exactly 30,000 updates.
