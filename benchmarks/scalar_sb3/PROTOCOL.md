# Frozen SB3 scalar-baseline protocol

## Research question

This controlled study tests whether the Scalar V2 instability is specific to
the historical SAC implementation. Stable-Baselines3 SAC and PPO receive the
same V2-B Balanced reward, physical task, training seeds, track splits, and
300,000-interaction budget.

The primary outcome is validation requirement satisfaction rate (RSR), computed
from `EpisodeMetrics` and `is_feasible`. Training return is diagnostic only.
Seeds 4000–4017 remain reserved and must not be generated or inspected.

## Common task and reward

The task is route completion within 140 seconds with exactly zero measured
speed excess, followed by feasible-only energy minimization. The frozen reward
is exactly Scalar V2-B Balanced:

```text
2.0 * delta_position / 1000 m
+ 2.0 on an on-time completion
- 0.5 * step_energy / 0.25 kWh
- 0.25 * delta_time / 140 s
- 0.1 * speed_excess * delta_time / 1 m
- 2.0 once at episode end if any speed excess occurred
```

No observation or reward normalization is applied. The historical eight-feature
observation space is already declared on `[0, 1]^8`; the reproducible empirical
audit is in `observation-audit.json`.

## Algorithm configurations

The recorded runtime uses Stable-Baselines3 2.9.0. All listed values are SB3
defaults and are explicit in `canonical.json` for provenance.

SAC uses learning rate `3e-4`, replay capacity `1,000,000`, 100 learning-start
interactions, batch size 256, `tau=0.005`, `gamma=0.99`, one update after every
environment step, automatic entropy tuning, and the default 256×256 ReLU policy
and critic networks.

PPO uses learning rate `3e-4`, rollout length 2,048, batch size 64, 10 epochs,
`gamma=0.99`, `gae_lambda=0.95`, clip range 0.2, entropy coefficient 0,
value coefficient 0.5, and the default 64×64 Tanh policy/value networks.

There is no algorithm-specific reward preprocessing. Architectures are not
artificially matched because the goal is a competent standard implementation,
not an ablation of network width.

## Exact sample-budget handling

External deterministic evaluation occurs at 50k, 100k, 150k, 200k, 250k, and
300k interactions on development tracks 2000–2008 and validation tracks
3000–3008. Evaluation uses a separate environment and does not enter replay or
rollout buffers.

PPO normally completes its 2,048-step rollout and can overshoot a requested
budget. The milestone callback instead stops exactly at 300,000 interactions.
Consequently, the last partial PPO rollout is not optimized; the final trained
policy reflects the last complete update at 299,008 interactions. Checkpoint
evaluations at other milestones likewise reflect the last complete PPO update.
This preserves the preregistered interaction-budget axis without changing
PPO's standard rollout length.

The budget does not imply equal compute: SAC updates after individual
interactions while PPO performs multiple epochs over rollout batches. Results
record both wall time and policy-update count.

## Predefined acceptance rule

The rule is inherited unchanged from Scalar V2. At 300k, every seed must reach
at least 8/9 RSR, 8/9 completion, and 8/9 speed compliance; the RSR range must
not exceed 1/9. On tracks where both are feasible, each seed's mean energy ratio
to the fast compliant oracle must not exceed 1.25. Severe late collapse also
precludes interpreting a method as a strong baseline.

## Predefined trajectory selection

Final diagnostics select the best- and lowest-RSR seed per algorithm and—when
present—the policy with the largest peak-to-final RSR decline. Ties use
completion and speed compliance, then the lower seed for best and higher seed
for worst. Tracks are selected by maximum common feasibility, maximum distinct
failure modes, and maximum progress among infeasible selected policies, with the
lowest track seed breaking ties.
