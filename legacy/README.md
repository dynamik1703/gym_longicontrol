# Historical implementation

`v0/` contains the original environment sources and model artifact, retained for
characterization tests and reproducing earlier work. It is not installed by the
modern package. The TensorFlow 1 DDPG code remains under `rl/ddpg/`, with its
original checkpoint and SHAP notebook under `Jupyter/`; these are historical
reference code, not supported training examples.

For the former four-value Gym API with the new simulator use the explicit
compatibility bridge documented in `MIGRATION.md`. Exact historical experiments
also require their original dependency versions and source checkout.
