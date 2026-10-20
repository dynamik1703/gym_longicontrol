# Frozen scalarization

## Prior source values

| Seed | lambda_speed at 300k | lambda_deadline at 300k |
|---:|---:|---:|
| 11 | 1.737065439763169 | 1.0015569080594906 |
| 29 | 1.4009421299572349 | 1.5752363435900516 |
| 47 | 1.3904230583041928 | 2.3950774369414662 |

Componentwise medians:

```text
lambda_speed_star    = 1.4009421299572349
lambda_deadline_star = 1.5752363435900516
Z                    = 3.9761784735472867
```

Final coefficients and reward:

```text
w_reward   = 0.2514977651664276
w_speed    = 0.35233381481173953
w_deadline = 0.39616842002183283

r_scalar(t) = 0.2514977651664276  * r_E(t)
            - 0.35233381481173953 * c_speed(t)
            - 0.39616842002183283 * c_deadline(t)
```

This is the primary frozen condition. It may not be changed in response to new policy
performance.

## Analytical scale sensitivity only

| multiplier scale | lambda_speed | lambda_deadline | w_reward | w_speed | w_deadline |
|---:|---:|---:|---:|---:|---:|
| 0.5x | 0.7004710649786174 | 0.7876181717950258 | 0.40191484502248825 | 0.2815297195236183 | 0.31655543545389336 |
| 1.0x | 1.4009421299572349 | 1.5752363435900516 | 0.2514977651664276 | 0.35233381481173953 | 0.39616842002183283 |
| 2.0x | 2.8018842599144698 | 3.150472687180103 | 0.14383611307786853 | 0.4030121412401577 | 0.4531517456819737 |

No sensitivity policy or hyperparameter sweep is authorized.

## Reward-scale audit

All 900,000 prior canonical TRANSITIONAL transitions were reconstructed without gaps
from the retained replay snapshots using `real_transition_id`. Pooled statistics:

| signal | mean | std | median | p90 | p95 | p99 | max abs | nonzero |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `r_E` | -0.000747723 | 0.00207358 | -0.000173868 | 0.00127697 | 0.00192517 | 0.00304635 | 0.00966893 | 1.000000 |
| `lambda_speed_star*c_speed` | 0.0140979 | 0.112010 | 0 | 0 | 0 | 0.461870 | 3.25099 | 0.0413622 |
| `lambda_deadline_star*c_deadline` | 0.0174042 | 0.0327652 | 0 | 0.0681534 | 0.0947650 | 0.136636 | 0.201923 | 0.350479 |
| `r_scalar` | -0.00811076 | 0.0290083 | -0.000624725 | 0.000280257 | 0.000450244 | 0.000744660 | 0.818633 | 1.000000 |

Per-seed values are frozen in `reward_scale.json`. The rare speed contribution has a
heavy tail but is finite and traceable to actual prior transition data. No catastrophic
numerical incompatibility is present, and no weight was changed after this audit.
