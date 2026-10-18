# Fingerprint specification

Schema ID: `longicontrol-content-fingerprint-v1`.

Both hashes are SHA-256 over UTF-8 canonical JSON. Mappings are key-sorted; sequences
retain order; finite floats are serialized with Python's exact hexadecimal
representation; NumPy arrays/scalars and dataclasses are reduced to values. NaN and
infinity, non-string mapping keys, and unsupported Python objects are rejected. Object
IDs, `repr`, insertion order, memory layout, and numeric seed are excluded.

## `route_sha256`

The route payload contains only realized operational infrastructure:

- total route length;
- cumulative segment-start positions, including zero;
- speed-limit transition positions;
- speed-limit sequence in m/s;
- ordered segment records with index, start, end, length, and limit;
- the 150 m sensor range, two-transition look-ahead, right-continuous limit semantics,
  and distance clamping used by observations.

The numeric RNG seed and unrealized random choices are excluded. Consequently two
different seeds that produce identical route arrays under identical infrastructure
semantics have the same route hash. Task requirements, rewards, costs, deadline,
energy factor, vehicle, and learning algorithm are also excluded from the route hash.

## `full_task_sha256`

The full-task payload contains the complete route payload and `route_sha256`, plus:

- simulation configuration: route length, native `dt=0.1 s`, sensor range;
- zero-valued initial vehicle state;
- vehicle mass, frontal area, drag, acceleration/velocity/power limits;
- power-model artifact SHA-256;
- air density, rolling resistance, gravity, acceleration interpolation grid;
- constant-acceleration integration, velocity clipping, energy conversion, and jerk
  conventions;
- observation fields and normalization, including `energy_factor`;
- stochastic generator support: transition sites/count rule, speed grid, adjacency
  rule, and initial-choice rule;
- the study-specific reward or objective;
- costs and their normalization;
- feasibility rule and every task requirement/deadline;
- physical and wrapper termination/truncation rules.

The study descriptor must explicitly supply `observation`, `dynamics`, `objective`,
`costs`, `feasibility`, `termination`, and `requirements`; omission is an error.
Algorithm settings that cannot change the environment/task outcome (network width,
optimizer, replay sampling, update cadence) are excluded. An algorithm-defined task
interface or objective (for example sparse goal reward or projected CRL objective) is
included. Training seed and policy parameters are excluded.

For fixed Constrained V2, route equality implies full-task equality because every
non-route task field is frozen. Requirement-conditioned tasks are different: the same
route with a different margin/deadline has the same route hash but a different
full-task hash.

## Correctness evidence

Tests establish that:

1. historical draw 316 and numeric seed 2006 hash identically;
2. changing one limit changes both hashes;
3. changing only energy factor or deadline preserves the route hash and changes the
   full-task hash;
4. repeated generation is stable;
5. mapping order and object identity do not affect serialization;
6. the split gate enforces seed-only, full-task-clean, and route-clean policies.

All fingerprint tests passed before results were interpreted.
