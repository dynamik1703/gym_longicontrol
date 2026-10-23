# Experiment Protocol

The frozen matrix has six independent runs: both methods at seeds 11, 29, and
47. Every run consumes exactly 300,000 real native transitions and 30,000
gradient updates from the inherited canonical seed-specific task stream. Arms
with the same seed use identical ordered tasks.

Revision-3 update accounting is authoritative:

```python
def cumulative_update_target(N):
    if N <= 10000:
        return 0
    return min(30000, ((N - 10000) * 30000) // 290000)
```

Replay insertion occurs only for completed/legitimately shortened macro
blocks. Technical checkpoints occur every 10k native steps, retained scientific
checkpoints every 50k, and heartbeats every 250. Diagnostic is evaluated only
at 100k, 200k, and the primary 300k endpoint. Checkpoint selection from outcomes
is forbidden.

Preflight uses only synthetic/unreserved fixtures and must pass parity, timing,
phase influence, unchanged gates, exact speed, newly visible transitions,
interruption, partial recovery, Bellman arithmetic, route cleanliness, finite
networks, and resource limits. Any failure stops before main training.

Diagnostic contains 36 new Policy-R tasks. Authorized comparators are CAUSAL60,
TIME_AWARE, FixedPrice MPC V2, AdaptivePrice MPC V2, and frozen Hybrid H11/H29/H47.
Each method-task pair is immutable and exactly once. Validation has 36 separately
frozen tasks but no authorized performance entry point. Paper remains sealed.

Feasibility precedes energy. Report raw RSR, completion, deadline, exact speed,
failure identities, feasible-only energy, paired per-route percentage saving,
travel time, intervention/influence, action structure, seed variation, critic
stability, resources, and integrity.

