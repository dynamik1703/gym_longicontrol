# Frozen Physics-only protocol

This protocol is frozen before any main-study transition. `canonical.json` is the
machine-readable authority.

## Question and matrix

> Does exact one-step physics-model augmentation improve the robustness of the frozen
> Constrained-V2 learner at the same real interaction budget?

The only condition is Physics-Dynamics MBRL, run serially for seeds 11, 29 and 47.
Each policy receives exactly 300,000 real native simulator transitions: 900,000 total.
There is no Learned Dynamics, residual model, MPC, planning, horizon sweep or ratio
sweep. Learned Dynamics is excluded because the prior mixed study established its
computational infeasibility before its preregistered endpoint. Its incomplete
Development results are not used to tune Physics.

## Byte-equivalent parent condition

The scientific implementation and parent configuration remain byte-for-byte frozen
and hash checked. Physics delegates to `domain.dynamics.advance` with known exogenous
track context. Training retains H=1, 10,000 real-step warmup, refresh every 250 real
steps, 2,500 synthetic transitions per refresh, and 128 real plus 128 synthetic rows
per mixed update. Real rows retain n-step 2; synthetic rows use the preregistered
one-step bootstrap. Objective, speed cost, deadline-deficit cost, PID, SACLag settings,
RNG streams and real-only PID updates are unchanged.

The new adapter changes only the manifest namespace and captures already-computed
logger/PID/cost values in checkpoint diagnostics. It does not change an update, target,
replay sample, action, RNG draw or simulator transition.

## Splits and execution

Training rejects tracks 1000-1008, 2000-2008, 3000-3008 and 4000-4017. Development
tracks 2000-2008 run at 50k, 100k, 150k, 200k, 250k and 300k. Validation tracks
3000-3008 open exactly once after all three 300k policies and their final hashes freeze.
Paper-test tracks 4000-4017 remain sealed.

Execution is detached and serial: Physics 11, 29, then 47, with four CPU threads.
Heartbeat, atomic full-state checkpoints every 10k, scientific checkpoints every 50k,
two-checkpoint technical retention, exclusive leases and exact resume are inherited.
No silent or automatic fresh restart is authorized. The new output root is
`runs/physics-model-mbrl-v1`; no mixed-study replay or checkpoint is reused.

## Preregistered interpretation

The prior mixed-study Physics Gate A is reused exactly:

- **Strong improvement:** RSR at least 22/27, speed compliance at least 24/27,
  completion 27/27 and deadline compliance 27/27.
- **No material improvement:** RSR at most 21/27.
- **Degradation context:** completion/deadline materially below frozen V2 or RSR
  clearly below 21/27. Raw deltas are always reported; no new post-result favorable
  threshold or significance claim will be invented.
- Values above 21 RSR that fail Gate A are reported as inconclusive between the two
  exact quantitative gates.

Frozen Constrained V2 is the historical reference: RSR 21/27, completion 27/27,
deadline 27/27 and speed 21/27. It is not retrained. Because Physics adds synthetic
H=1 targets while V2 is all-real n-step 2, this is a close historical comparison, not
a perfect one-factor causal control.

## Evaluation and reporting

External `EpisodeMetrics` and benchmark feasibility are authoritative. Validation is
3 policies x 9 tracks = 27 real episodes with deterministic learned actors and the
original Gymnasium simulator. There is no model query, planning or synthetic
interaction at evaluation. Feasible-only energy is null if no episode is feasible.

Report per-seed and pooled RSR; completion, deadline and speed compliance; exclusive
failure modes; Development curves; feasible-only mean/median energy; real/synthetic/
update accounting; speed/deadline costs; multipliers; actor/critic losses; source-wise
TD errors; Physics parity and wall-clock.

## Runtime and authorization

The frozen resource probe estimates 1.1 CPU-serial hours per Physics policy and 3.3
hours for the serial matrix. This estimate is descriptive, not a contractual limit.
Device (`cpu`) and thread count (`4`) remain fixed.

Main training and the post-freeze one-shot Validation are authorized. Paper-test and
automatic fresh restart are not authorized.
