# Capability Results

| Arm | RSR | Complete | Deadline | Speed | Energy feasible kWh | Time s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| COMMIT_FEEDBACK | 18/18 | 18 | 18 | 18 | 0.176739850 | 97.244 |
| CONTINUE_FEEDBACK | 18/18 | 18 | 18 | 18 | 0.172722352 | 97.200 |
| FROZEN_MPC_V2 | 18/18 | 18 | 18 | 18 | 0.171096061 | 97.228 |
| RESET_FEEDBACK | 18/18 | 18 | 18 | 18 | 0.171765449 | 98.550 |

Primary CONTINUE versus RESET jointly feasible count:
18; arithmetic mean per-task saving:
-0.5620832885179694; median:
-0.6053189838917894; range: [-2.304587674150191, 2.982112623446665].

Scientific classification: `SEMANTIC_OR_INTEGRITY_STOP`. This label
concerns a non-learning execution architecture and is not an RL result.

The fail-closed label is caused by missing retention of per-step completion
bounds, not by a changed or rerun Pair. Before the integrity correction, the
automatic performance label was `OPTIONS_VALID_EFFECT_UNCLEAR`: all arms were
18/18 feasible, but CONTINUE used 0.5621% more energy than RESET on the arithmetic
mean of paired task percentages. See `INTEGRITY_INCIDENT.md`.
