# Migrating from 0.0.1 to 1.0.0

Version 1.0 modernizes LongiControl around the Gymnasium API and separates the
simulation model from rendering and training.  The physical integration,
eight-dimensional normalized observation, action range, and legacy reward
formula are retained away from the velocity-boundary correction described below.

## Package and imports

Install the environment itself with:

```bash
python -m pip install .
```

Rendering and the bundled training example are optional:

```bash
python -m pip install ".[render]"
python -m pip install ".[train]"
```

Replace `gym` with `gymnasium`:

```diff
- import gym
+ import gymnasium as gym
  import gym_longicontrol
```

The new environment IDs are `DeterministicTrack-v1` and
`StochasticTrack-v1`.  The `-v1` suffix makes the API break explicit; the old
`-v0` registrations are not silently redirected.

For a transition period, a four-value adapter is available without old Gym:

```python
from gym_longicontrol.compat import make_legacy_env

env = make_legacy_env("DeterministicTrack-v0")
env.seed(2)
observation = env.reset()
observation, reward, done, info = env.step([0.5])
env.close()
```

Existing Gym integrations can install `.[legacy]` (Gym 0.23.1), call
`gym_longicontrol.compat.register_legacy_envs()`, and use
`gym.make("DeterministicTrack-v0")`. Use explicit registration and the bare ID;
Gym 0.23's module-prefix lookup is inconsistent. Both adapters emit a
deprecation warning. They preserve the API shape, not all v0 numerical quirks.

## Reset and seeding

Seeding moved from `env.seed(...)` to `env.reset(seed=...)`, and reset now
returns an observation and an info mapping:

```diff
- env.seed(2)
- observation = env.reset()
+ observation, info = env.reset(seed=2)
```

Calling `reset()` without a seed continues the environment's random-number
stream.  Calling it again with the same seed reproduces a stochastic track.
Gymnasium uses NumPy's Generator, so a v0 seed does not reproduce the identical
historic track sequence. Store track arrays when comparing experiments across
versions. The historical sources remain under `legacy/v0/`.

## Stepping and episode completion

Gymnasium distinguishes a task terminal state from a wrapper time limit:

```diff
- observation, reward, done, info = env.step(action)
+ observation, reward, terminated, truncated, info = env.step(action)
+ done = terminated or truncated
```

Reaching the end of the 1,000 m track sets `terminated`.  The registered
environment's 1,800-step time limit sets `truncated`.

The info mapping retains the short historic keys and adds names with explicit
units.  New integrations should use the explicit keys such as `position_m`,
`velocity_m_s`, `acceleration_m_s2`, and `total_energy_kwh`.

## Rendering

Choose rendering when constructing the environment.  `render()` no longer
accepts a mode:

```diff
- env = gym.make("gym_longicontrol:DeterministicTrack-v0")
- env.render(mode="rgb_array")
+ env = gym.make(
+     "gym_longicontrol:DeterministicTrack-v1",
+     render_mode="rgb_array",
+ )
+ frame = env.render()
```

Headless training does not import rendering dependencies.
The optional renderer now uses Matplotlib and plots speed limits, speed, and
acceleration. The previous Pyglet dashboard and its keyboard shortcuts remain
in the archived implementation. Install `.[video]` to use video recording.

## Explicit numerical corrections

The old environment declared physical bounds while returning normalized
observations. Version 1.0 declares the actual normalized eight-feature space.
It also limits acceleration at within-step velocity boundary crossings so the
next velocity stays in [0, 37] m/s; v0 could slightly overshoot its stated
maximum. Deterministic rollout regression tests compare the archived equations
with the new implementation away from this boundary, and a separate test
checks the corrected boundary behavior. Invalid/non-finite inputs now raise
descriptive exceptions rather than relying on assertions.

## Reward and `energy_factor`

The signed reward components are now exposed as `info["reward_components"]`.
Their sum after applying `reward_weights` is the scalar reward.

For numerical compatibility, `energy_factor` remains the eighth observation
feature.  It does **not** multiply the energy reward in 1.0.0; use the energy
entry in `reward_weights` to change that term.  This corrects the ambiguous
description in the former training CLI without changing trained-policy input
semantics.

## Vehicle power artifact

Runtime loading no longer unpickles a scikit-learn 0.21 estimator.  The fitted
layer weights are stored in a small NumPy archive and evaluated by the core
package.  The conversion is reproducible with
`tools/convert_legacy_power_model.py`; its JSON metadata records the source
artifact hash, original sklearn version, feature order, units, and layer
shapes.

## Training and checkpoints

Run `python -m training` from a repository checkout after installing `.[train]`.
The wheel installs only the environment; the trainer remains example code in
the checkout. Training now uses the complete [-1, 1] initialization range,
corrects the replay-buffer first-write index, and bootstraps time-limit
truncations. These fixes intentionally change learning results.

New checkpoints include the target network, optimizer state, replay buffer,
random-generator state, and resolved run configuration. CPU continuation at an
epoch boundary is tested against an uninterrupted next update. Cross-device
bitwise reproducibility is not promised. Resume with the same environment,
network, step-count, replay and optimizer configuration.

New history files are JSON, next to `seedN.config.json`. The former
`rl/pytorch/monitor.ipynb` reads only historic `.npy` histories. Use
`python -m training.monitor path/to/seedN.json` for new runs.

New checkpoints use PyTorch's restricted tensor loader by default. Loading the
original trusted `.tar` artifacts may require `--trust_legacy_checkpoint`;
they lack replay/RNG state, so continuation cannot reproduce the original run
exactly. Keep the legacy flag off for files whose source you do not trust.

For example, load the bundled historical policy with:

```bash
python -m training --env_id DeterministicTrack-v1 --visualize \
  --checkpoint rl/pytorch/out/DeterministicTrack-v0/SAC_id9/seed2.tar \
  --trust_legacy_checkpoint
```

## Legacy research code

The TensorFlow 1 DDPG implementation and its SHAP notebook are historical
artifacts.  They are kept for reference but are not installed, tested, or
pulled in as package dependencies.  The supported training path uses the
Gymnasium-compatible PyTorch code documented in the README.
