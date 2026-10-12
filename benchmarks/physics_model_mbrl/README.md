# Physics-Dynamics Model-Based Constrained RL

This separately preregistered study asks whether exact one-step physics-model
augmentation improves the robustness of the frozen Constrained V2 learner at the
same real interaction budget. It runs only Physics-Dynamics MBRL for seeds 11, 29
and 47. Each policy receives 300,000 real simulator transitions.

The controller remains a learned SACLag actor. The model is queried during training
only; evaluation uses no planning and is not MPC.

The study descends from termination commit
`e0a6b7a2068eb0e0f9917831ff420fd7bd10aa9a`. The mixed Learned-vs-Physics study and
its artifacts remain frozen. Learned Dynamics is absent because its computational
infeasibility was established before the preregistered endpoint, not because of a
performance result.

`canonical.json` and `PROTOCOL.md` define the study. Raw checkpoints, replay and logs
live under the ignored `runs/physics-model-mbrl-v1/` root. Compact parity and final
results are tracked here.

Useful commands:

```bash
python -m benchmarks.physics_model_mbrl.runner preflight
python -m benchmarks.physics_model_mbrl.runner status
python -m benchmarks.physics_model_mbrl.parity
python -m benchmarks.physics_model_mbrl.detached_launcher launch
python -m benchmarks.physics_model_mbrl.runner validate
```
