# Constrained V2 seed-11 track × collector factorial V2

This repaired study stopped at its mandatory route-clean preflight. No policy was
trained or evaluated.

The frozen diagnostic task `9000005` occurs naturally in the exact `T_CURRENT`
training stream at zero-based episode index 559, where explicit reset seed
`1141367263` produces the same canonical route and full task. The collision is
inside the largest physically possible stream prefix for a 300,000-transition run.
Replacing, filtering, or resampling that task or the training stream is forbidden by
the protocol, so the main matrix is not authorized.

The exact fastest bounded-action trajectory needs 347 native steps per route. Thus
at most 864 episodes can complete in 300,000 transitions and one additional task can
be active: 865 task exposures must be checked. `T_HIST` is clean across those 865
tasks; `T_CURRENT` is not.

Evidence is in [results.json](results.json), with the zero-execution state in
[manifest.json](manifest.json) and frozen file hashes in [freeze.json](freeze.json).
`preflight.py` regenerates the evidence without calling `env.step`, evaluating a
policy, or constructing any Paper-final task.
