"""Offline CAPQL / scalarized SAC comparison. Run from a repository checkout."""

import argparse
import hashlib
import json
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import numpy as np

from examples.morl.evaluation import (
    DEFAULT_REFERENCE,
    DEFAULT_SCALE,
    DEFAULT_WEIGHTS,
    ENV_IDS,
    evaluate,
    make_env,
    validate_weights,
)
from gym_longicontrol.envs.multi_objective import REWARD_NAMES

ALGORITHMS = ("sac-sweep", "capql")


@dataclass(frozen=True)
class BenchmarkConfig:
    env_id: str = ENV_IDS[0]
    algorithms: tuple = ALGORITHMS
    steps: int = 50_000
    train_seeds: tuple = (42, 43, 44)
    eval_seeds: tuple = (1001, 1002, 1003)
    weights: tuple = DEFAULT_WEIGHTS
    max_episode_steps: int = 1800
    learning_starts: int = 1000
    batch_size: int = 64
    reference: tuple = DEFAULT_REFERENCE
    scale: tuple = DEFAULT_SCALE

    def __post_init__(self):
        validate_weights(self.weights)
        if self.env_id not in ENV_IDS:
            raise ValueError(f"Unknown environment: {self.env_id}")
        if not self.algorithms or not set(self.algorithms) <= set(ALGORITHMS):
            raise ValueError(f"algorithms must be selected from {ALGORITHMS}")
        if len(set(self.algorithms)) != len(self.algorithms):
            raise ValueError("Algorithms must be distinct")
        for name in ("steps", "max_episode_steps", "learning_starts", "batch_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.learning_starts < self.batch_size:
            raise ValueError("learning_starts must be at least batch_size for CAPQL")
        if self.batch_size > 50_000:
            raise ValueError(
                "batch_size must not exceed the replay buffer size (50000)"
            )
        count = len(self.weights) if "sac-sweep" in self.algorithms else 1
        if self.steps // count <= self.learning_starts:
            raise ValueError("The per-policy budget must exceed learning_starts")
        for name in ("train_seeds", "eval_seeds"):
            seeds = getattr(self, name)
            if (
                not seeds
                or len(set(seeds)) != len(seeds)
                or any(
                    not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in seeds
                )
            ):
                raise ValueError(f"{name} must contain distinct integers in [0, 2**32)")
        if set(self.train_seeds) & set(self.eval_seeds):
            raise ValueError("Training and evaluation seeds must not overlap")
        reference, scale = np.asarray(self.reference), np.asarray(self.scale)
        if (
            reference.shape != (4,)
            or scale.shape != (4,)
            or not np.isfinite(reference).all()
            or not np.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ValueError(
                "reference and positive scale must have four finite values"
            )


def split_budget(total, count):
    """Distribute the entire budget; never silently multiply it by policy count."""
    quotient, remainder = divmod(total, count)
    return [quotient + (index < remainder) for index in range(count)]


def _dependencies():
    try:
        import torch
        from mo_gymnasium.wrappers import LinearReward
        from morl_baselines.multi_policy.capql.capql import CAPQL
        from stable_baselines3 import SAC
    except ImportError as error:
        raise ImportError(
            'MORL examples require: python -m pip install -e ".[morl]". '
            "See examples/morl/README.md for native dependencies."
        ) from error

    class OfflineCAPQL(CAPQL):
        def close_wandb(self):
            # Upstream calls this unconditionally, even with log=False. Do not
            # finish a caller's unrelated W&B session in this offline example.
            if self.log:
                super().close_wandb()

    return torch, LinearReward, OfflineCAPQL, SAC


def _write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _artifact(path, root):
    return {
        "path": str(path.relative_to(root)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _train(algorithm, seed, config, directory, dependencies):
    """Use upstream implementations; no algorithm copies or global monkeypatches."""
    torch, linear_reward, capql, sac = dependencies
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    common = dict(
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        buffer_size=50_000,
        batch_size=config.batch_size,
        learning_starts=config.learning_starts,
        seed=seed,
        device="cpu",
    )
    weights = validate_weights(config.weights)
    environments = []
    models, artifacts = [], []
    try:
        if algorithm == "capql":
            train_env = make_env(
                config.env_id, max_episode_steps=config.max_episode_steps
            )
            environments.append(train_env)
            eval_env = make_env(
                config.env_id, max_episode_steps=config.max_episode_steps
            )
            environments.append(eval_env)
            # Upstream train() calls reset() without a seed. Seed the environment
            # RNG beforehand, as well as the warm-up action sampler.
            train_env.reset(seed=seed)
            train_env.action_space.seed(seed)
            eval_env.reset(seed=config.eval_seeds[0])
            model = capql(train_env, net_arch=[64, 64], alpha=0.2, log=False, **common)
            model.train(
                total_timesteps=config.steps,
                eval_env=eval_env,
                ref_point=np.asarray(config.reference),
                checkpoints=False,
            )
            model.save(
                save_dir=str(directory), filename="capql", save_replay_buffer=False
            )
            artifacts.append(directory / "capql.tar")
            policies = [lambda obs, w=w: model.eval(obs, w) for w in weights]
            actual_steps = int(model.global_step)
        else:
            budgets = split_budget(config.steps, len(weights))
            for index, (weight, budget) in enumerate(zip(weights, budgets)):
                env = linear_reward(
                    make_env(config.env_id, max_episode_steps=config.max_episode_steps),
                    weight=weight,
                )
                environments.append(env)
                model = sac(
                    "MlpPolicy",
                    env,
                    policy_kwargs={"net_arch": [64, 64]},
                    train_freq=1,
                    gradient_steps=1,
                    ent_coef=0.2,
                    verbose=0,
                    **common,
                )
                model.learn(total_timesteps=budget)
                path = directory / f"sac_{index}.zip"
                model.save(path)
                models.append(model)
                artifacts.append(path)
            policies = [
                lambda obs, m=m: m.predict(obs, deterministic=True)[0] for m in models
            ]
            actual_steps = sum(model.num_timesteps for model in models)
        if actual_steps != config.steps:
            raise RuntimeError(
                f"Training budget mismatch: {actual_steps} != {config.steps}"
            )
        evaluation = evaluate(
            policies,
            weights=weights,
            seeds=config.eval_seeds,
            env_id=config.env_id,
            max_episode_steps=config.max_episode_steps,
            reference=config.reference,
            scale=config.scale,
        )
        return {
            "algorithm": algorithm,
            "training_seed": seed,
            "training_steps": actual_steps,
            "steps_per_policy": (
                [config.steps]
                if algorithm == "capql"
                else split_budget(config.steps, len(weights))
            ),
            "artifacts": [_artifact(path, directory.parent) for path in artifacts],
            **evaluation,
        }
    finally:
        for env in environments:
            env.close()


def plot_results(runs, path):
    """Pairwise projections of 4-D mean returns; not claimed 2-D Pareto fronts."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 10))
    FigureCanvasAgg(figure)
    colors = {"capql": "tab:orange", "sac-sweep": "tab:blue"}
    for ax, (left, right) in zip(figure.subplots(3, 2).flat, combinations(range(4), 2)):
        labelled = set()
        for run in runs:
            points = np.asarray([row["mean_return"] for row in run["policies"]])
            name = run["algorithm"]
            ax.scatter(
                points[:, left],
                points[:, right],
                color=colors[name],
                alpha=0.65,
                label=name if name not in labelled else None,
            )
            labelled.add(name)
        ax.set(
            xlabel=f"{REWARD_NAMES[left]} return (maximize)",
            ylabel=f"{REWARD_NAMES[right]} return (maximize)",
        )
        ax.legend()
    figure.suptitle("Mean returns per policy and training seed (pairwise projections)")
    figure.tight_layout()
    figure.savefig(path)


def run(output, config=BenchmarkConfig(), *, plot=True):
    dependencies = _dependencies()
    torch = dependencies[0]
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "objective_names": REWARD_NAMES,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: version(name)
            for name in (
                "gym-longicontrol",
                "gymnasium",
                "numpy",
                "torch",
                "stable-baselines3",
                "morl-baselines",
                "mo-gymnasium",
                "pymoo",
                "matplotlib",
                "setuptools",
            )
        },
        "source_base_revision": revision,
        "source_sha256": {
            str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(
                [
                    Path(__file__),
                    *repo.joinpath("examples/morl").glob("*.py"),
                    *repo.joinpath("src/gym_longicontrol").rglob("*.py"),
                    *repo.joinpath("src/gym_longicontrol/assets/vehicle").glob("*"),
                ]
            )
        },
        "training_settings": {
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "net_arch": [64, 64],
            "buffer_size": 50_000,
            "entropy_coefficient": 0.2,
            "gradient_updates_per_step": 1,
            "device": "cpu",
            "torch_threads": 1,
            "external_logging": False,
        },
        "limitations": (
            "Integration example, not an optimized benchmark or safety claim. "
            "All four objectives are maximized. Hypervolume uses undiscounted "
            "mean returns and is computed separately per training seed. "
            "Timeouts are included: always inspect completion and violations. "
            "Energy is predicted; forward is speed-limit tracking, not travel time; "
            "shock is overspeed, not collision safety. "
            "The true Pareto front is unknown. "
            "Upstream CAPQL samples preferences near uniform, not the full simplex. "
            "Checkpoints are for trusted local use, not exact training resume."
        ),
    }
    _write_json(output / "config.json", report)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        runs = []
        for seed in config.train_seeds:
            for algorithm in config.algorithms:
                directory = output / f"{algorithm}-seed-{seed}"
                directory.mkdir()
                print(
                    f"{algorithm}: seed={seed}, total training steps={config.steps}",
                    flush=True,
                )
                result = _train(algorithm, seed, config, directory, dependencies)
                _write_json(directory / "evaluation.json", result)
                runs.append(result)
        report["runs"] = runs
        report["summary"] = {
            name: {
                "training_seeds": len(config.train_seeds),
                "hypervolume_mean": float(np.mean(values)),
                "hypervolume_std": float(np.std(values)),
            }
            for name in config.algorithms
            for values in [[r["hypervolume"] for r in runs if r["algorithm"] == name]]
        }
        _write_json(output / "evaluation.json", report)
        if plot:
            plot_results(runs, output / "return_projections.png")
        return report
    finally:
        torch.set_num_threads(previous_threads)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--algorithms", choices=ALGORITHMS, nargs="+", default=ALGORITHMS
    )
    parser.add_argument("--env-id", choices=ENV_IDS, default=ENV_IDS[0])
    parser.add_argument(
        "--steps",
        type=int,
        default=50_000,
        help="Total steps per algorithm and training seed",
    )
    parser.add_argument("--train-seeds", type=int, nargs="+", default=(42, 43, 44))
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=(1001, 1002, 1003))
    parser.add_argument(
        "--weight", type=float, nargs=4, action="append", dest="weights"
    )
    parser.add_argument("--max-episode-steps", type=int, default=1800)
    parser.add_argument("--learning-starts", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reference", type=float, nargs=4, default=DEFAULT_REFERENCE)
    parser.add_argument("--scale", type=float, nargs=4, default=DEFAULT_SCALE)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)
    options = vars(args).copy()
    output, no_plot = options.pop("output"), options.pop("no_plot")
    options["weights"] = options["weights"] or DEFAULT_WEIGHTS
    for key in ("algorithms", "train_seeds", "eval_seeds", "reference", "scale"):
        options[key] = tuple(options[key])
    options["weights"] = tuple(tuple(w) for w in options["weights"])
    try:
        config = BenchmarkConfig(**options)
    except ValueError as error:
        parser.error(str(error))
    report = run(output, config, plot=not no_plot)
    print(json.dumps(report["summary"], indent=2))
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
