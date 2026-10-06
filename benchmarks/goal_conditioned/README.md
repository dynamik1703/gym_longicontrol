# Goal-conditioned SAC and HER

This benchmark area contains the completed matched comparison between
goal-conditioned Stable-Baselines3 SAC with zero virtual goals and the same
learner with constraint-preserving Hindsight Experience Replay.

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
- `runner.py`: duplicate-protected execution, checkpoint and Validation gate;
- `analysis.py`: integrity-checked compact result generation;
- `plot_results.py`: reproducible plots from compact result data;
- `RESULTS.md` and `results.json`: final scientific result and derived data;
- `validation_episodes.json`: the 54 final external-evaluator episodes;
- `execution_manifest.json`: compact execution provenance and model hashes.

The public `-v1` environments, historical Binary V1 implementation and all
completed study artifacts remain unchanged. Raw models and checkpoints remain
under the ignored `runs/` directory; only compact results and provenance are
tracked here. Reserved paper-test tracks 4000--4017 remain unopened.
