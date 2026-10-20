# Source and Information Audit

## Authoritative simulator

The design keeps the production `StochasticTrack-v1` path unchanged:

- `LongiControlEnv.step` clips one native action to `[-1, 1]`, calls production
  `advance`, observes the route at the new right endpoint and updates production
  `EpisodeMetrics`.
- `advance` uses the exact frozen BMW i3 action/acceleration and power model,
  0.1 s constant-acceleration dynamics and signed net-energy accumulation.
- `Track.sense` returns the current limit plus at most two transitions inside
  the 150 m preview. It does not reveal the remainder of the track.
- Completion is 1,000 m. Deadline and exact zero speed tolerance remain external
  `TaskSpecification` / `is_feasible` requirements.

The source authority is the physical-reference base commit
`3d9e8bcb72ecf4a01be192036df8f00c5ac9de8e`. The frozen scientific
configuration hash is
`f63717504a810c75d7080388aeb159505cf667ad13248dbe4a53ec0c4bb94cd1`.
All six physical-reference source hashes are rechecked by `prepare.py` and are
recorded in `preparation_status.json`.

## Exact online information boundary

The future Hybrid policy may receive only:

1. the ordinary normalized 8D sensor observation: velocity, acceleration,
   current speed limit, two visible future limits, their distances and the
   historical energy-factor field;
2. controller-propagated position and elapsed time;
3. remaining distance and remaining time derived from fixed task constants;
4. the previous executed native action and previous target speed;
5. the causal deadline-bound slack;
6. the current frozen-fallback proposal;
7. the known exact vehicle/dynamics, 140 s deadline, 1,000 m length, 150 m
   preview and generator bounds.

These form the exact 16D strategy observation in `controller.py`. The fallback
continues to remember speed transitions only after they become visible and to
propagate its physical memory solely from executed actions.

The two-stage call is:

```text
prepare_strategy_step(ordinary_observation) -> 16D causal strategy input
strategy policy -> scalar action in [-1, 1]
action(same ordinary observation, strategy action) -> shielded native action
observe_executed_action(native action) -> advance controller memory
```

The API accepts no environment object, `Track`, `info` dictionary, simulator
`VehicleState`, Oracle trajectory/action, checkpoint-specific task metadata or
future state. Static AST tests reject `.track` and `.unwrapped` access in the
Hybrid module.

## Fallback integration

`ObservationLimitedController.action` is inherited unchanged. When the shield
rejects a proposal, that exact action is applied. When a proposal is accepted,
the adapter supplies the actually executed action to the same exact-dynamics
memory-propagation method. It does not change braking logic, margins or the
60 km/h fallback cap.

With influence disabled, the adapter delegates to an unmodified
`ObservationLimitedController`; six deterministic fixtures match every action,
state trace and final metric byte-for-byte at the numeric-value level. Hashes
and per-fixture step counts are in `preflight_results.json`.

## Provenance and independence

The target-speed choice, 20–60 km/h bounds, 2 s deadline reserve, algorithm,
training budget, split namespaces and success criteria are frozen in
`canonical.json`. They are design assumptions grounded in the completed
physical-reference study and task semantics. The newly computed post-hoc
contextual outcomes were not used for controller selection, reward selection,
hyperparameters or split selection.

No training runner, policy checkpoint loader, behavior cloning path, Oracle
demonstration path or distillation code exists in this package.
