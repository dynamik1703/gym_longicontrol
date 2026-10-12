# Mixed MBRL study termination

The study was terminated because measured computational cost made the
preregistered six-policy experiment operationally infeasible. The termination was
not triggered by Development performance.

The decision was recorded at 2026-10-12 17:52:35 Europe/Berlin
(15:52:35 UTC). The running Learned-Dynamics refresh was allowed to finish. A
controlled `SIGINT` was then sent at the next complete execution boundary, at
2026-10-12 18:16:21 Europe/Berlin. The interrupt reached the following policy
forward pass before another environment transition. The worker's context manager
recorded the interrupt and removed the active lock; the serial launcher stopped and
did not start another policy. The inactive one-shot LaunchAgent was booted out at
2026-10-12 18:22:59 Europe/Berlin so it cannot run again automatically; its plist
and logs remain preserved.

## Resource evidence

The preparation estimate was 94.2 serial CPU-hours for all six policies. Measured
wall time increased to 15.1 hours from 40k to 50k real transitions and 18.9 hours
from 50k to 60k. Attempt 3 ran for 52.81 wall-clock hours. Its final observed worker
CPU time was at least 56.41 hours. At the measured rate, the first Learned policy
alone had hundreds of hours remaining and the serial matrix was projected at roughly
1,600--2,500 additional hours.

## Preserved state

- Last verified recovery checkpoint: 60,000 real transitions, SHA-256
  `155fa64e06de2bdd9848a6a9f4535f144c0d790fe0de1f19f9eadc3a47d00dd7`.
- Durable counters: 6,000 RL updates, 29,123,065 model-gradient updates, 201
  model refreshes, 502,500 generated and 640,128 sampled synthetic transitions.
- Last complete observed boundary: 60,500 real transitions, 6,050 RL updates,
  29,700,838 model-gradient updates, 203 refreshes, 507,500 generated and 646,528
  sampled synthetic transitions. The additional work after 60k is explicitly
  recorded as observed but not durable.
- Attempts 1, 2 and 3 remain preserved. Attempt 3 is classified
  `TERMINATED_RESOURCE_INFEASIBLE`, not as a policy or scientific failure.

## Partial scientific information

The only completed Development evaluation was the preregistered 50k checkpoint on
tracks 2000--2008. It produced 0/9 completions and 0/9 feasible episodes, with 9/9
speed-compliant episodes and mean final position 0 m. This result is retained as
partial evidence but was not the termination criterion.

The verified 60k checkpoint contains 201 learned-model training reports and 5,001
mixed-update diagnostic rows; all stored numeric diagnostic values are finite. The
last refresh used 60,000 real examples, 5,000 holdout examples, seven 200-epoch
members and five elites. No main-study false-safe diagnostic was persisted, so Gate E
is `NOT_AVAILABLE` and must not be inferred from these data.

Learned-Dynamics MBRL was not evaluated at the preregistered 300k endpoint because
the measured compute cost rendered the frozen experiment infeasible. No final RSR is
reported or extrapolated.

The other Learned seeds and all Physics seeds remained `NOT_STARTED`. Validation and
paper-test tracks remained `SEALED`. No scientific model, objective, cost, ratio,
threshold, task or Development datum was changed. Machine-readable evidence is in
`termination.json`; raw logs and verified checkpoints remain in the ignored run
directory.
