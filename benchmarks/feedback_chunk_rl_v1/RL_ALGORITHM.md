# RL Algorithm

Both arms use Gaussian-tanh constrained SAC with two 256-unit ReLU layers,
twin reward critics, separate speed and terminal-infeasibility critics, and an
intervention critic used with fixed weight 0.05. Target-network `tau` is 0.005,
batch size 256, actor/critic/entropy learning rates are `3e-4`, and multiplier
learning rate is `1e-3`. Speed and terminal limits are zero and multipliers are
projected nonnegative.

Target entropy is minus the action dimension: -3 for CHUNK3 and -1 for
RESIDUAL1S. Actor output means are deterministically initialized near zero and
initial log standard deviation is frozen at `log(0.2)`. The warm-up samples the
same bounded zero-centered distribution; it does not issue random native
actions. No model, imitation loss, privileged critic, sweep, or post-Diagnostic
tuning is permitted.

The reward is `-1000 * signed_step_energy_kWh`. Positive endpoint speed excess,
terminal failure, and native intervention deviation remain separate signals.
The intervention signal compares the combined native actuator proposal with
the executed native action.

