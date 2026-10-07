# Primary-source audit

Audit date: 2026-10-07. The requested, immutable paper target is
[Scaling CRL v4](https://arxiv.org/html/2503.14858v4) (2026-02-02), even though
arXiv showed a later v5 on the audit date. The foundational method source is
[Contrastive Learning as Goal-Conditioned RL](https://arxiv.org/abs/2206.07568).
The official implementation is pinned to
[`17acb519ddc4325c8662b1f8c68ed6a5f31857fc`](https://github.com/wang-kevin3290/scaling-crl/commit/17acb519ddc4325c8662b1f8c68ed6a5f31857fc).
Hashes are in `upstream.json`; no moving branch is needed to reproduce the audit.

| Original method | Planned LongiControl implementation | Reason for deviation |
|---|---|---|
| State-action encoder receives `(s, a)`; goal encoder receives achieved future state coordinates `g`. | State-action encoder receives the 12-value sensor/task state plus action. Goal encoder receives a 3-value requirement-aware projection of an actually recorded raw future outcome. | Raw position, previous position, absolute time, and cumulative violation remain in replay/state; the projection is a benchmark-specific task abstraction, not the unchanged paper representation. |
| Actor receives state concatenated with a goal. | Collection/evaluation command `[1,1,1]`; actor updates receive sampled projected future outcomes from replay. | This freezes the pinned scaling code's future-goal actor update while allowing an unobserved canonical command. Collection and training goal distributions differ explicitly. |
| Four `Dense -> LayerNorm -> Swish` operations per residual block; skip addition after the fourth activation. Depth counts Dense layers inside blocks only. | Same architecture, width 256, embedding 64; input and output projections reported separately. Actor and both encoders scale together. | No architectural deviation. |
| Published code uses `q=-sqrt(sum((phi-psi)^2))` and minimizes `alpha log pi - q`. | Same negative-distance sign, with `+1e-8` inside the square root. | The guard gives finite gradients at identical embeddings. Paper v4 equation 4 displays a positive norm, which would reverse the implemented actor direction if read literally. |
| Diagonal future pairs are positives; every in-batch goal column is a reference class. | Same diagonal-positive InfoNCE; duplicates are measured, not silently masked. | Multi-positive masking would be a methodological change and is not introduced during preparation. |
| Future index is strictly later in the same Brax episode/seed and sampled proportional to `gamma^lag`, `gamma=.99`. | Strictly later in the same physical episode and uninterrupted collector segment, with the same per-decision weighting. | Explicit episode and step IDs prevent pairs crossing resets or discarded collector transitions. |
| Upstream `buffer.py` samples complete 1000-step sequences, identifies episode continuity through Brax seed metadata, and drops the last source row. | Backend-neutral trajectory sampling interface; no fixed 1000-step assumption. | LongiControl remains the existing CPU Gymnasium simulator and is not ported to JAX. |
| Foundational paper derives future positives from discounted occupancy and uses reference-marginal goals; it reports random/reference actor goals outperforming future actor goals. Pinned scaling code instead feeds sampled future goals to the actor update. | Freeze the pinned code choice: projected sampled futures train the actor; `[1,1,1]` commands real collection and evaluation. | This is an implementation-versus-foundational-paper difference, not silently merged or performance-tuned. |
| Actor/alpha update occurs before critic update. Target entropy is `-0.5 * action_dim`; learned alpha starts at one. | Same order, target, initialization, and Adam learning rates (`3e-4`). | No deviation. |
| Critic objective is diagonal InfoNCE plus `0.1 * mean(logsumexp(logits + 1e-6)^2)`. | Same equation. | No deviation. |
| Simulator rewards are stored but unused by actor and critic losses. | `historical_reward` may be retained in batches but is never read; equality-of-loss/gradient tests perturb it. | Makes absence of reward leakage auditable. |
| Code defaults: batch 256, depth 4, width 256, replay size 10,000 **time rows × 512 envs**, minimum 1,000 rows, 800 minibatches after each 62-step × 512-env collection. README example uses batch 512/depth 16; paper table reports batch 512. | Freeze batch 256 and one complete update cycle per 40 post-prefill native transitions. | Batch follows the pinned code default; the 512 difference is disclosed. The serial schedule matches aggregate sample use but is a LongiControl adaptation. |
| Upstream scale is 100M transitions by default (paper experiments 100M–400M), 512 parallel envs, and large GPU execution. | Draft tier is only 300k native transitions per policy on the unchanged CPU simulator. | This is a low-budget adaptation, not a replication; a failure cannot refute scaling results. |

## Exact implemented equations

For embeddings `z_sa = phi(s,a)` and `z_g = psi(g)`:

```text
C(s,a,g) = -sqrt(||z_sa - z_g||² + 1e-8)
L_NCE = -mean_i [ C_i,i - logsumexp_j C_i,j ]
L_critic = L_NCE + 0.1 mean_i [ logsumexp_j(C_i,j + 1e-6)² ]
L_actor = mean[ exp(log_alpha) log pi(a|s,g) - C(s,a,g) ]
L_alpha = exp(log_alpha) mean(stop_gradient(-log pi - target_entropy))
target_entropy = -0.5 * action_dim
```

`C` is an uncalibrated contrastive association/density-ratio score. It is not
RSR, a first-arrival probability, or a hard constraint certificate.

## Task-projection revision

The verified loss, optimizer order, temporal weights, reference columns, and
residual architecture are unchanged. Only goal/actor input dimension changes
from four exact outcome coordinates to three projected coordinates. Projection
collapses exact outcomes according to the sampling-induced projected reference
distribution; no arbitrary physical-goal integration, calibrated probability
head, duplicate mask, or alternate NCE variant is introduced.

## Dependencies and licensing

Upstream requires Python 3.10 and pins NumPy 1.26.4, JAX/JAXlib 0.4.23,
Flax 0.7.4, Brax 0.10.1, SciPy 1.12.0, MuJoCo 3.2.6, W&B 0.17.9, and related
packages. This preparation needs only the isolated subset in
`requirements-reference.txt`; it deliberately omits Brax, MuJoCo, W&B, and
telemetry. The upstream code is Apache-2.0, Copyright 2023 FLAIR. Provenance,
modification notice, and license text are retained locally.
