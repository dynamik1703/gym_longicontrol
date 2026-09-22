# MORL-Baselines integration

This checkout-only example compares **CAPQL (MORL-Baselines 1.3.0)** with an
**SB3 SAC weight sweep** on continuous actions. It is an integration starting
point, not a tuned benchmark. Importing the example does not train anything.
No model download, W&B account, external logging, GUI or GPU is needed.

## Installation

The MO environments need only core NumPy/Gymnasium. Training and reporting
require the optional `morl` extra. Run from a checkout or extracted source
distribution: wheels intentionally do not install `examples`.

MORL-Baselines 1.3.0 requires `pycddlib==2.1.6` even for CAPQL. Building this
dependency requires a C compiler and GMP headers when no wheel is available.
On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential libgmp-dev
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[morl]"
python -m pip check
```

On macOS with Homebrew, Xcode Command Line Tools and Python 3.11 or 3.13:

```bash
HOMEBREW_NO_INSTALL_CLEANUP=1 brew install gmp
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
CFLAGS="-I$(brew --prefix gmp)/include" \
LDFLAGS="-L$(brew --prefix gmp)/lib" \
  python -m pip install -e ".[morl]"
python -m pip check
```

MORL CI targets Python 3.10 and 3.13 on Linux. Local macOS integration tests
also pass with Python 3.11 and 3.13. On our macOS 27 test system, Python 3.10's
SciPy 1.15.3 wheel fails to load its PROPACK extension (`__thread_bss` zero-fill
offset error) before CAPQL can import. A matching issue is reported in
[SciPy #25635](https://github.com/scipy/scipy/issues/25635). Use a fresh Python
3.11/3.13 environment on macOS; the tested versions are SciPy 1.17.1 (Python
3.11) and 1.18.1 (Python 3.13).
Do not patch compiled libraries or disable runtime validation to bypass this.
Python 3.10 remains in the Linux CI matrix. Native Windows builds of pycddlib
are not claimed as tested. Core/scalar installations do not need GMP or SciPy.
Setuptools supplies upstream's `distutils` import on Python versions where it
is no longer in the standard library.

## Environment contract

```python
import gymnasium as gym
import gym_longicontrol

env = gym.make("MOStochasticTrack-v1")
observation, info = env.reset(seed=42)
observation, reward, terminated, truncated, info = env.step([0.5])
assert reward.shape == (4,)
assert env.unwrapped.reward_space.contains(reward)
assert env.unwrapped.reward_dim == 4
print(env.unwrapped.reward_names, reward)
env.close()
```

`MODeterministicTrack-v1` is also available. Both reuse scalar v1 dynamics,
observations, configuration, info fields and the 1,800-step time limit.
Existing scalar IDs and legacy adapters are unchanged. All signed objectives
are **maximized**, in this fixed order:

| Index / name | Per-step meaning |
| --- | --- |
| 0 / `forward` | Negative absolute speed-limit tracking error divided by the current limit. **Not travel time**. |
| 1 / `energy` | Negative predicted power divided by the upper power limit. May be positive during regeneration. **Not kWh**. |
| 2 / `jerk` | Negative absolute acceleration change divided by the acceleration range. |
| 3 / `shock` | -1 above the speed limit, otherwise 0. **Not collision safety**. |

Objectives are not redefined or rescaled. `reward_weights` is rejected: apply
preferences in the agent or `mo_gymnasium.wrappers.LinearReward`.
`energy_factor` remains observation context, not a preference or energy-reward
multiplier. CAPQL receives the full preference via `agent.eval(obs, weight)`.

The interface follows [MO-Gymnasium's contract](https://mo-gymnasium.farama.org/tutorials/custom_env/).
`mo_gymnasium.make`, `LinearReward` and `MORecordEpisodeStatistics` are tested.
Only MO IDs disable Gymnasium's scalar-reward passive checker; dedicated tests
validate their vector contract instead.

## Run

Installation smoke test (**not a performance result**):

```bash
python -m examples.morl_baselines --output runs/morl-smoke \
  --steps 125 --train-seeds 42 --eval-seeds 1001 \
  --learning-starts 8 --batch-size 8 --max-episode-steps 8
```

Full episodes with several independent training seeds:

```bash
python -m examples.morl_baselines --output runs/morl-comparison \
  --steps 50000 --train-seeds 42 43 44 --eval-seeds 1001 1002 1003
```

`--steps` is the **total training interaction budget per algorithm and training
seed**: by default 10,000 steps for each of five SAC policies, versus 50,000 for
one preference-conditioned CAPQL agent. Remainders are distributed explicitly.
Evaluation interactions are separate and equal in number. SAC is a
scalarization baseline, not another dedicated MORL algorithm. Equal interaction
budgets do not imply equal compute or tuned hyperparameters.

Both use CPU, one PyTorch thread, [64, 64] networks, gamma 0.99 and fixed entropy
coefficient 0.2. Python/NumPy/PyTorch RNGs and training environments are seeded
explicitly. Evaluation uses deterministic actions on identical requested
seeds in fresh environments. Test episodes never enter replay or model selection.
Training/evaluation seed lists must be disjoint. Bit-for-bit reproduction across
platforms or package upgrades is not promised.

Options: `--algorithms capql`, `--algorithms sac-sweep`, `--no-plot`,
`--env-id MODeterministicTrack-v1`, or repeated `--weight 0.4 0.2 0.2 0.2`
arguments. Each preference must sum to 1. Different seeds on the deterministic
track do not represent independent track samples.

Upstream CAPQL samples preferences in a 22.5-degree cone around uniform, not
the full simplex. Defaults are uniform plus four nearby vectors with one
weight 0.4 and the others 0.2. Custom weights change CAPQL **evaluation**, not
its upstream training sampler. Performance at extreme/unseen weights is not
assured. Algorithm code is not copied or globally monkeypatched. A small local
subclass prevents upstream's unconditional `wandb.finish()` from closing an
unrelated W&B session when `log=False`; learning code is unchanged. See
[upstream CAPQL](https://github.com/LucasAlegre/morl-baselines/blob/main/morl_baselines/multi_policy/capql/capql.py).

## Results and limitations

Each invocation requires a new output directory; existing data is not overwritten:

- `config.json`: configuration, versions, source hashes/base revision, objective
  order, training settings and limitations.
- `evaluation.json`: per-episode physical metrics, vector returns, policy means,
  approximation fronts and hypervolume; mean/population std across training
  seeds, not confidence intervals.
- `return_projections.png`: six pairwise projections of four-dimensional mean
  returns. These are not asserted two-dimensional Pareto fronts.
- Per-algorithm/seed directories: evaluations and native model artifacts
  (`capql.tar`, `sac_*.zip`); SHA-256 hashes appear in the report.

First average **undiscounted** vector returns across test tracks per policy.
Compute nondominated sets/hypervolume separately per training seed, never from
a union across training seeds. All objectives are maximized. For metrics only,
divide returns and the fixed reference by a positive scale vector. Defaults:
`--reference -12000 -2000 -2000 -2000`, `--scale 1800 1800 1800 1800`.
The scale is fixed, not episode length. Training uses original rewards and
gamma 0.99. A return below the reference raises an error; choose a common
reference before rerunning all compared methods. Changing objectives, horizon,
reference or scale makes reports incomparable unless all methods are rerun
consistently.

Timeouts remain in the report. **Inspect completion and overspeed alongside
rewards**: barely moving can appear to save energy. Energy is predicted, not
measured. The true Pareto front is unknown: no IGD, Pareto-optimality,
algorithm-superiority or validated-safety claim is made.

Upstream loaders can deserialize pickle objects. Load only trusted artifacts;
hashes verify integrity, not safety. CAPQL saves without replay history; exact
training resume is not promised. Reload your own CAPQL artifact by constructing
CAPQL with `net_arch=[64, 64]`, `log=False`, the same environment/configuration,
then calling `agent.load(path, load_replay_buffer=False)`. Use `SAC.load` for SAC.

## Towards upstream inclusion

No upstream inclusion or endorsement is implied. Before proposing a listing
or example in MORL-Baselines / MO-Gymnasium:

1. Merge/release the API and integration with passing CI.
2. Publish full-budget multi-seed results and limitations.
3. Clarify the project license; provide stable installation, a minimal standalone
   example and citation. This integration does not assign a license to the repo.
4. Discuss the appropriate contribution with upstream maintainers.

For research, cite LongiControl (root README),
[MORL-Baselines / its toolkit paper](https://github.com/LucasAlegre/morl-baselines#citing-the-project)
and the algorithm paper where applicable.
