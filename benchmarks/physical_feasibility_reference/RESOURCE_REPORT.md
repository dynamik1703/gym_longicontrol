# Resource Report

The preflight measured three Development routes before final evaluation.

| Quantity | Result | Gate |
| --- | ---: | ---: |
| Wall time, 3 trajectories | 0.9298 s | — |
| Mean initial trajectory | 0.3099 s | — |
| Projected full execution | 200.83 s (0.0558 h) | ≤12 h: pass |
| Projected retained artifacts | 0.003456 GB | ≤12 GB: pass |
| Measured maximum RSS | 55,377,920 platform units | recorded |
| Checkpoint overhead | none | deterministic controllers |

The completed Development and Holdout panels took 37.41 s and 71.52 s,
respectively. The final namespace, including 54 persisted Oracle witnesses, is
approximately 3.3 MB. No checkpoint was created or modified.

Mean/maximum per-task Oracle solve times on holdout were 1.8553/3.6655 s.
Mean/maximum causal-controller episode times were 0.1242/0.2058 s. Every
resource gate passed without weakening safety semantics.
