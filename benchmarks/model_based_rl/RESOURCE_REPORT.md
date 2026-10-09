# Resource and preparation report

The committed machine-readable measurements are `resource_measurements.json`,
`physics_parity.json`, `learned_model_probe.json` and
`preparation_interactions.json`.

On the recorded Apple ARM CPU, the learned ensemble has 862,456 parameters (3.45 MB of
raw parameter storage); the frozen SACLag policy/critics have 232,974 parameters. Exact
physics projection measured about 457 microseconds per transition and sampled learned
projection about 545 microseconds. Learned synthetic generation measured roughly 1,834
transitions/s. A pinned FSRL/Tianshou 50/50 mixed update probe performed 14.81 gradient
updates/s. Policy plus ensemble serialized to 4.45 MB before replay; the three full
100k replay payloads have a 48 MB array lower bound, so full checkpoints are expected
above roughly 52 MB plus container overhead.

The bounded model-disabled integration used 512 analytically generated transitions and
zero simulator interactions. One direct frozen `policy.update` and one call through the
model-disabled adapter produced bit-identical parameters (maximum difference zero),
retained n-step 2 and left both PID multipliers at zero. This verifies implementation
parity; it is not a retrained 300k V2 control.

Linear CPU extrapolation gives about 30.3 hours for one Learned policy and 1.1 hours for
one Physics policy, or about 94.2 serial CPU hours for all six. The Learned estimate is
dominated by repeatedly fitting all seven members and is conservative; it is not a
promise. No paid/cloud hardware was provisioned.

The bounded Learned probe used 396 targeted examples for training and 99 held out. Its
purpose was numerical and instrumentation verification, not a quality gate or future
initialization. It produced finite NLL, five elites and all requested metrics. The
small, deliberately difficult probe had 10/99 false-safe cases, 5/99 false-unsafe,
velocity RMSE 0.0445 m/s, position RMSE 0.624 m and energy RMSE 0.000510 kWh. This is a
warning that justifies the preregistered Gate E; the future main ensemble starts empty,
trains after 10k ordinary real samples and is not initialized from this probe.

Preparation consumed 4,995 of the maximum 5,000 simulator transitions, all on
Development seeds. The first 4,500-transition computation completed but its final
worktree artifact write was sandbox-denied; those interactions remain conservatively
counted. It was not rerun. The remaining targeted 495-transition probe supplied the
committed aggregates. Validation and paper transitions are both zero.
