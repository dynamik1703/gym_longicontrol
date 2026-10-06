# Goal-conditioned SAC and HER

This benchmark area prepares a matched comparison between goal-conditioned
Stable-Baselines3 SAC with zero virtual goals and the same learner with
constraint-preserving Hindsight Experience Replay. No policy training result is
included yet.

The canonical task is still completion of the original route by 140 seconds
with zero speed excess. Real rollouts always use that goal. HER broadens only
the training replay task family by substituting an intermediate route position;
it does not relabel the deadline, speed tolerance, elapsed time or accumulated
violation history.

Files:

- `DESIGN.md`: physical-state audit and exact goal/replay semantics;
- `PROTOCOL.md`: frozen future experiment;
- `canonical.json`: machine-readable configuration;
- `goal.py`: vectorized encoding, success, reward and terminal helpers;
- `environment.py`: benchmark-only Dict observation adapter;
- `replay_buffer.py`: shared episode-complete replay and minimal HER adapter;
- `experiment.py`: environment/model factories that do not start training.

The public `-v1` environments, historical Binary V1 implementation and all
completed study artifacts remain unchanged. A future execution stage must add a
duplicate-run-protected runner; this preparation commit deliberately provides
no command that launches the 1.8-million-transition study.
