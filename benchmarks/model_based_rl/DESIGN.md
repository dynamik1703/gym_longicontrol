# Design

## Question and causal contrast

The study asks whether explicit one-step model-based imagination improves requirement
satisfaction and robustness when added to the best existing constrained formulation.
The two conditions differ only in the source of vehicle-dynamics predictions:

- **Learned-Dynamics MBRL with known exogenous track map**;
- **Physics-Dynamics MBRL**, an oracle/known-model condition.

Both final controllers are ordinary SACLag actors. Models run only during training;
there is no planning, CEM, trajectory optimization or MPC at evaluation time.

## State boundary and observability

The public policy observation remains the original eight normalized values: velocity,
acceleration, current limit, two future limits, two distances and energy factor. It has
no position, elapsed time, cumulative safety history or full route map. In particular,
a previously invisible limit can enter the 150 m sensor window after one transition.
The observation is therefore not a Markov state for exact next-observation prediction.

The benchmark-internal `ModelState` contains:

```text
VehicleState(position, velocity, acceleration, jerk, elapsed time, total energy)
maximum speed excess
contiguous speed-violation count and active-event flag
integrated speed violation
episode step
```

`TrackContext` contains the immutable speed-limit map, route length, sensor range and
energy factor. It is known exogenous context supplied to both models and never supplied
to the actor. The learned model input is only:

```text
[position_m, velocity_m_s, acceleration_m_s2, action]
```

It predicts a Gaussian distribution over:

```text
[delta_position_m, delta_velocity_m_s,
 next_acceleration_m_s2, signed_step_energy_kwh]
```

Elapsed time advances exactly by 0.1 s. Jerk, total energy, track sensing, public next
observation, objective, both costs, speed history, completion and timeout are computed
by one deterministic projection shared by both conditions. Energy is learned in the
Learned condition because substituting the exact power model would leak the vehicle
dynamics under test.

## Physics model

`PhysicsDynamicsModel` calls the same pure `domain.dynamics.advance` function as the
Gymnasium environment. It does not reimplement an approximate equation. The shared
projection then uses the same observation function, V2 objective and V2 costs.

The preregistered absolute tolerance is `1e-12` for physical quantities, observations
and costs, plus exact terminal flags. The targeted 495-transition recovery probe spans
all Development track seeds, 450 speed-limit boundary crossings, acceleration, braking
and route-end cases. Every criterion passed; the largest error was floating-point speed
excess at `3.55e-15` m/s. Timeout semantics are covered separately by a deterministic
unit test.

## Learned ensemble

Seven independent four-layer 200-unit Swish networks each emit mean and bounded log
variance for four normalized targets. Five elites are selected by member holdout NLL.
At every refresh a seeded permutation creates a 20% holdout set (maximum 5,000), and
each member receives a seeded bootstrap of the training set. Adam uses `1e-3`; maximum
training is 200 epochs; member parameters revert to their best holdout state after more
than five epochs without at least 1% relative improvement. Input and target statistics,
all model/optimizer states, elites and RNG state are checkpointed. Only real
transitions enter this path.

## Imagination and learner data

No synthetic transition is generated before 10,000 real transitions. Thereafter both
conditions refresh every 250 new real transitions and generate 2,500 H=1 transitions.
Each starts from a uniformly sampled real-replay state and uses an action drawn from the
current stochastic actor. Learned predictions sample one seeded elite and its Gaussian;
Physics predictions are deterministic. Provenance includes source, condition, source
real-transition ID, model version and ensemble member.

A bounded model-disabled test feeds identical analytical batches and RNG state through
the direct frozen V2 update and the adapter. The resulting parameters are bit-identical,
n-step remains two and PID remains untouched; see `model_disabled_parity.json`.

RL optimizer count remains `floor(0.1 * real transitions)`, exactly 30,000 at 300k.
Before warmup, updates are real-only. Afterwards every 256-sample update contains 128
real and 128 synthetic samples. Synthetic generation never increments the real budget.

FSRL requires contiguous replay indices for its frozen two-step return. Real samples
retain that exact path. An H=1 synthetic branch has no legitimate second transition, so
it uses a one-step bootstrapped target with the same gamma and target critics. This
difference is fixed for both model conditions and is not tuned. Model-disabled parity
uses the untouched all-real two-step path.

## Constraint semantics

The objective and ordered costs remain:

```text
objective = -signed_step_energy_kwh / 0.25
cost[0] = max(0, next_velocity - next_speed_limit) * 0.1
cost[1] = 0.1 * max(0, -(140 - next_elapsed - T_min(next_position))) / 140
```

No margin, heuristic, progress reward, bonus or changed limit exists. Only a completed
real episode calls FSRL's PID `pre_update_fn`; a synthetic episode raises an error. Thus
model bias can affect critics and actor, but never directly manipulate multipliers.

## Diagnostics

The implementation records continuous one-step errors, compliant/violating confusion,
false-safe and false-unsafe rates, speed-excess magnitude error, deadline-cost category,
termination, ensemble variance, elite identities, holdout loss, epochs, real examples,
refreshes, synthetic generated/sampled, model version, source fractions and source-wise
TD errors. Open-loop errors at 1/5/10/25/50 are descriptive only and cannot change H=1.
