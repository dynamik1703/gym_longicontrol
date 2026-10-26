# Source audit

Repository: dynamik1703/gym_longicontrol. Verified push remote
`gym_longicontrol_origin` = `https://github.com/dynamik1703/gym_longicontrol.git`.
The unrelated Trading Dashboard origin was inspected and never used.
Branch `research/bankers-viability-shield-v2`, worktree
`/Users/romanliessner/gitlab/gym_longicontrol_bankers_shield_v2`.
Base commit `0162477848bcfef2d1acea3e38a47b8fbd539299`; preflight implementation
committed as `ccb8c2771d4c94be7125422ca683751d5a9a8ceb` before scientific fixtures.

All 4,624 inherited tracked files match that base commit. Their individual
SHA-256 values are in source_audit.json, including every V1 artifact, production
module, BMW power artifact and historical research package. V2 imports the V1
certificate, causal memory, backup and menu unchanged; only evidence/controller
adaptation and independent verification are new. No physics/generator, sensor,
reward, feasibility or termination source is modified.

Audited production: domain/dynamics.py, state.py, vehicle.py, observation.py,
track.py, task.py, metrics.py and envs/longicontrol.py. Native zero is real
coasting; energy is signed power*0.1/3600; exact endpoint speed excess is measured
independently from reward. is_feasible applies inclusive 140 s and zero excess
without epsilon. TimeLimit is 1800 and the original finish is position>=1000.
The StochasticTrack generator separates signs by >=100 m, limits are 20..100 km/h
and adjacent changes are <=40 km/h. The complete two-slot sensor reaches 150 m.

Sources and ordinary-data channels were audited in the original V1 reports
and inspected directly against the frozen production implementation. Online
controllers receive ordinary observation and causal action-propagated state only.
Route access in V2 is confined to static split selection, synthetic fixtures,
post-execution evidence collection and offline replay.

V2 identity (outer records bind this separately from unchanged inner V1 identities):

```json
{
  "base_commit": "0162477848bcfef2d1acea3e38a47b8fbd539299",
  "configuration_sha256": "3b505e32d18f7d9d8dd72e831e0510477b16062f0a2af71e4136448277fb3e79",
  "contract": "bankers-evidence-v2.0",
  "implementation_sha256": "a04b0b4599e666a3a2fded2192b0dbc14c3edbcbf2f6e970fcd10bf5952d37f2",
  "inherited_tree_sha256": "b7051efaa433bf9e0116afa31b5c271d3d917a9dfbdc7bbb22ce4649d6d9bb89"
}
```

The implementation identity includes V2 Python/test files and fixed proof,
serialization and protocol documents; configuration hashes canonical.json.
The inherited-tree identity hashes the full audit file, whose ordered inherited
content digest is `e808ed9c9b9732f3e7f322ed254bc68617904f57bdf4e4c27f95c2c5d09d391d`.
Preflight runtime is the existing Python 3.11.13 environment with NumPy 2.4.6
and Gymnasium 1.4.0, with PYTHONPATH=src:. resolving this worktree production code.
