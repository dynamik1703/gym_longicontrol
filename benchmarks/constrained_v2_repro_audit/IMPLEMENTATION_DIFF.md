# End-to-end implementation diff

## Training timelines

Historical:

```text
collect complete episode with fixed policy
  -> exact episodic cost vector
  -> PID pre_update_fn
  -> round(0.1 * episode steps) consecutive policy.update calls
  -> checkpoint/evaluate crossed targets
```

Current Real-Only:

```text
collect one transition
  -> record physical sidecar state
  -> if terminal: PID pre_update_fn
  -> catch up to floor(total real transitions * 0.1) policy.update calls
  -> checkpoint/evaluate exact targets
```

For a 1,800-step first episode, the deterministic schedule fixture places all 180
historical learner updates after PID. The current actor first changes at transition
256, has completed 179 updates before the terminal PID event, and performs the 180th
update after PID on the terminal step. Therefore the policy collecting later portions
of the same current episode is not the policy that collected its beginning.

## Collector and transition semantics

| Property | Historical | Current | Finding |
|---|---|---|---|
| Collector | FSRL `FastCollector` | Tianshou `Collector` | Different implementation |
| Collection unit | one complete episode | one transition | Material schedule difference |
| Actor changes mid-episode | No | Yes after replay warm-up | Material |
| Terminal reset | in-loop reset plus final episode-mode reset | in-loop reset only | Material track-RNG difference |
| Replay class/capacity | Tianshou `ReplayBuffer`, 100,000 | same | Identical |
| Replay insertion fields | obs/action/reward/done/next obs/vector cost | same | Fixed-policy synthetic episode hash matches |
| Terminated/truncated storage | both retained in current Tianshou data schema | same | No mismatch found |
| Budget terminal | dense wrapper truncates at 300,000 total steps | same wrapper | Same task signal; surrounding schedule differs |

A three-step, fixed-policy synthetic fixture produces byte-identical transition
payload hashes for both collectors. This isolates the issue: storage semantics match
when input trajectories match, but real trajectories and call cadence do not.

## N-step and replay

Both paths ultimately call the same pinned FSRL SACLagrangian
`compute_nstep_returns` with `n_step=2`, gamma 0.99, one reward critic and two ordered
cost critics. The deterministic fixture verifies:

- identical replay sampling for identical buffer plus NumPy state;
- successor indices stop at episode boundaries;
- order is reward, speed cost, deadline cost;
- done/value masking is common;
- effective two-step return tensor hash is stable.

Thus no n-step formula mismatch was found. In real execution, the formula receives
different buffers, indices, policy targets, and RNG states from the first episode.

## PID/Lagrange timing

Both pipelines pass the exact undiscounted completed-episode vector in order
`[speed_integral_m, deadline_deficit_integral_s]` to the same FSRL
`pre_update_fn`. `post_update_fn` is a no-op in the pinned FSRL code. No cost-order or
PID-gain mismatch was found.

The timing and counts differ:

| Seed | Historical PID events | Current PID events | Historical final speed/deadline λ | Current final speed/deadline λ |
|---:|---:|---:|---|---|
| 11 | 249 | 222 | 2.84770 / 1.21534 | 2.14137 / 1.66205 |
| 29 | 240 | 230 | 1.82546 / 1.15648 | 1.40813 / 1.08678 |
| 47 | 245 | 240 | 1.87505 / 1.17156 | 1.35084 / 0.98455 |

Counts differ because learned behavior produces different episode lengths; ordering
also differs independently because historical PID precedes a whole episode's updates
while current PID follows nearly all updates accrued during that episode. The final
multiplier shift is consistent with different trajectories/scheduling, but it is not
standalone proof that PID caused the Validation gap.

## Alpha/entropy

Both constructors use automatic entropy tuning, alpha 1.0, target entropy -1, alpha
LR `3e-4`, the same squashed-probability policy/log-prob implementation, and one alpha
update per learner update. No source/default mismatch was found. Final alpha differs:

| Seed | Historical | Current |
|---:|---:|---:|
| 11 | 0.0004161 | 0.0045278 |
| 29 | 0.0007945 | 0.0009871 |
| 47 | 0.0007257 | 0.0004619 |

Those values are consequences of different batches, actor states, update order, and
slightly different total update counts rather than evidence of a changed alpha rule.

## M=0 parity: what was and was not proved

For the current codebase, `real_replay_update(model_steps=0)` directly delegates to
the untouched pinned `policy.update()` and adds no synthetic target. That is genuine
**within-current-code policy-update parity**.

It does not establish **historical end-to-end training parity**. Operations outside
`policy.update()` differ: environment seed construction, reset count, collection
granularity, actor-update interleaving, PID ordering, update rounding/count, runtime,
checkpointing, and evaluation cadence.

## Recovery, diagnostics, and sidecar

All three current Real-Only runs completed in one attempt without resume. The physics
sidecar records transitions but does not alter the M=0 replay distribution. Model
rollout and diagnostic RNGs are separately constructed; M=0 makes no model target.
No evidence shows recovery or diagnostic code directly consumed the actor/replay RNG
before the first divergence. These facts do not restore parity because the dedicated
track stream and runner cadence already differ.
