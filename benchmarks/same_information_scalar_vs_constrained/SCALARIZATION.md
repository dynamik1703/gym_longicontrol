# Frozen scalarization and entropy-scale audit

## Reference-informed multipliers

| Seed | lambda_speed at 300k | lambda_deadline at 300k |
|---:|---:|---:|
| 11 | 1.737065439763169 | 1.0015569080594906 |
| 29 | 1.4009421299572349 | 1.5752363435900516 |
| 47 | 1.3904230583041928 | 2.3950774369414662 |

The componentwise medians remain unchanged:

```text
lambda_speed_star    = 1.4009421299572349
lambda_deadline_star = 1.5752363435900516
Z                    = 1 + lambda_speed_star + lambda_deadline_star
                     = 3.9761784735472867
```

No performance observation entered the following choice.

## Candidates

Candidate N, the original preparation, normalized only the immediate scalar task
reward while retaining standard adaptive-alpha SAC:

```text
r_N = (r_E - lambda_speed_star*c_speed
           - lambda_deadline_star*c_deadline) / Z
L_N = alpha*log pi - Q_N
```

If `Q_N` scaled ideally as `Q_U/Z`, multiplying this actor loss by `Z` exposes its
relative weighting:

```text
Z*L_N = Z*alpha*log pi - Q_U
```

At equal alpha, Candidate N therefore gives entropy `Z = 3.9761784735472867` times
the weight it has relative to the unnormalized task-value term in Candidate U.

Candidate U keeps completely standard SAC and uses the direct fixed Lagrangian
immediate reward:

```text
r_U = r_E
    - 1.4009421299572349*c_speed
    - 1.5752363435900516*c_deadline

L_U = alpha*log pi - Q_U
```

Candidate E would keep `r_N` but try to use an effective `alpha/Z`. With a fixed
temperature and ideal `Q_N=Q_U/Z`, this makes its whole actor objective `L_U/Z`,
which has the same instantaneous minimizer as `L_U`. It is not a clean matched
automatic-alpha condition here. The pinned alpha optimizer learns `log_alpha` from
only `log pi + target_entropy`; it contains no reward or `Z`. Initializing effective
alpha at `1/Z` would break the retained `alpha=1`, `log_alpha=0` initialization, while
post-scaling a learned alpha or coupling it to another run would introduce custom
temperature semantics. Policy-dependent alpha adaptation also cannot guarantee a
persistent `1/Z` relationship after the policies diverge. Candidate E is rejected
and is not implemented.

## Positive scaling: optimum versus dynamics

For fixed critics, multipliers, and alpha, multiplying the entire Constrained actor
objective by positive `Z` does not change its instantaneous minimizer. Its numerator
is:

```text
alpha*log pi - Q_r + lambda_speed*Q_speed + lambda_deadline*Q_deadline
```

and FSRL divides that complete expression by `Z`. The division does change gradient
magnitude and hence finite-step optimization dynamics under the frozen optimizer and
learning rate (including Adam's epsilon, moment history, and the changing objective).

Candidate U is nevertheless the clean standard-SAC control: its task reward is the
direct fixed Lagrangian combination and its entropy coefficient matches the
Constrained numerator. Candidate N instead normalizes the task signal without
normalizing the standard SAC entropy term, adding an avoidable task-versus-entropy
confound. Candidate U is therefore frozen as primary.

At initialization, both Candidate N and U have the same retained actor and reward
critic bytes and `alpha=1`; before a reward-conditioned critic update their actor
terms are byte-identical. As learning incorporates rewards, U's immediate task signal
and ideal task-value scale are exactly `Z` times N's, while both alpha optimizers use
the same reward-independent equation. Automatic alpha may react indirectly as their
policies change, but that feedback does not make the learning processes equivalent.

## Analytical lambda sensitivity only

The selected convention is unnormalized, so the reward coefficient remains `1.0`
and only the two fixed penalty coefficients vary:

| multiplier scale | reward coefficient | speed coefficient | deadline coefficient |
|---:|---:|---:|---:|
| 0.5x | 1.0 | 0.7004710649786174 | 0.7876181717950258 |
| 1.0x | 1.0 | 1.4009421299572349 | 1.5752363435900516 |
| 2.0x | 1.0 | 2.8018842599144698 | 3.150472687180103 |

No sensitivity policy, tuning run, or hyperparameter sweep is authorized. Primary is
exactly `1.0x`.

## Immutable-data reward-scale audit

All 900,000 prior canonical TRANSITIONAL transitions were reconstructed without gaps
from the six retained replay snapshots per seed using `real_transition_id`. No
environment, policy, collector, or evaluator was run. Full per-seed statistics and
both candidates are in `reward_scale.json`.

Candidate U is exactly `Z` times Candidate N elementwise. This audit is solely a
numerical sanity check; no historical return or policy outcome was used to select U.

Pooled statistics:

| signal | mean | std | median | p90 | p95 | p99 | min | max | max abs | nonzero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `r_E` | -0.000747723 | 0.00207358 | -0.000173868 | 0.00127697 | 0.00192517 | 0.00304635 | -0.00966893 | 0.00403285 | 0.00966893 | 1.000000 |
| `lambda_speed*c_speed` | 0.0140979 | 0.112010 | 0 | 0 | 0 | 0.461870 | 0 | 3.25099 | 3.25099 | 0.0413622 |
| `lambda_deadline*c_deadline` | 0.0174042 | 0.0327652 | 0 | 0.0681534 | 0.0947650 | 0.136636 | 0 | 0.201923 | 0.201923 | 0.350479 |
| `r_N` | -0.00811076 | 0.0290083 | -0.000624725 | 0.000280257 | 0.000450244 | 0.000744661 | -0.818633 | 0.00101425 | 0.818633 | 1.000000 |
| `r_U` | -0.0322498 | 0.115342 | -0.00248402 | 0.00111435 | 0.00179025 | 0.00296090 | -3.25503 | 0.00403285 | 3.25503 | 1.000000 |
