"""Train or try a small Stable-Baselines3 SAC agent from a repository checkout."""

import argparse
import copy
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

ENV_IDS = ("StochasticTrack-v1", "DeterministicTrack-v1")
DEMO_DIR = Path(__file__).resolve().parent / "models" / "sac_demo"
DEFAULT_SEEDS = tuple(range(1001, 1006))
SAC_SETTINGS = {
    "learning_rate": 3e-4,
    "buffer_size": 50_000,
    "batch_size": 64,
    "gamma": 0.99,
    "tau": 0.005,
    "train_freq": 1,
    "gradient_steps": 1,
    "ent_coef": "auto_0.1",
    "policy_kwargs": {"net_arch": [64, 64]},
}


def make_env(env_id=ENV_IDS[0], *, render_mode=None):
    """Use the registered time limit and unchanged v1 reward/observations."""
    import gymnasium as gym

    if env_id not in ENV_IDS:
        raise ValueError(f"Unknown environment: {env_id}")
    return gym.make(f"gym_longicontrol:{env_id}", render_mode=render_mode)


def _sb3():
    try:
        from stable_baselines3 import SAC
    except ImportError as error:
        raise ImportError(
            'This example requires: python -m pip install -e ".[examples]"'
        ) from error
    return SAC


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    """Never overwrite an existing report or checkpoint metadata."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def rollout(model=None, *, env_id=ENV_IDS[0], seed=1001, render=False):
    """One complete episode; None selects a seeded uniform-random policy."""
    env = make_env(env_id, render_mode="human" if render else None)
    try:
        observation, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        trajectory = []
        total_reward = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = (
                env.action_space.sample()
                if model is None
                else model.predict(observation, deterministic=True)[0]
            )
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            trajectory.append(
                {
                    key: float(info[key])
                    for key in (
                        "elapsed_time_s",
                        "position_m",
                        "velocity_km_h",
                        "speed_limit_km_h",
                        "acceleration_m_s2",
                        "jerk_m_s3",
                        "total_energy_kwh",
                    )
                }
            )
        metrics = {
            "seed": int(seed),
            "return": float(total_reward),
            "completed": bool(terminated),
            "truncated": bool(truncated),
            "steps": len(trajectory),
            "distance_m": float(info["position_m"]),
            "elapsed_time_s": float(info["elapsed_time_s"]),
            "energy_kwh": float(info["total_energy_kwh"]),
            "mean_abs_jerk_m_s3": float(
                np.mean([row["jerk_m_s3"] for row in trajectory])
            ),
            "overspeed_time_s": sum(
                row["velocity_km_h"] > row["speed_limit_km_h"] for row in trajectory
            )
            * env.unwrapped.config.dt_s,
        }
        return {"metrics": metrics, "trajectory": trajectory}
    finally:
        env.close()


def evaluate(model=None, *, env_id=ENV_IDS[0], seeds=DEFAULT_SEEDS):
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    episodes = [rollout(model, env_id=env_id, seed=seed)["metrics"] for seed in seeds]
    keys = (
        "return",
        "distance_m",
        "elapsed_time_s",
        "energy_kwh",
        "mean_abs_jerk_m_s3",
        "overspeed_time_s",
    )
    return {
        "env_id": env_id,
        "episodes": episodes,
        "summary": {
            "episodes": len(episodes),
            "completion_rate": float(np.mean([row["completed"] for row in episodes])),
            **{
                (key if key.startswith("mean_") else f"mean_{key}"): float(
                    np.mean([row[key] for row in episodes])
                )
                for key in keys
            },
        },
    }


def compare(model, *, env_id=ENV_IDS[0], seeds=DEFAULT_SEEDS):
    seeds = tuple(seeds)
    return {
        "schema_version": 1,
        "sac": evaluate(model, env_id=env_id, seeds=seeds),
        "random": evaluate(env_id=env_id, seeds=seeds),
        "limitations": (
            "Illustrative comparison, not a benchmark: one training seed, five "
            "evaluation seeds by default. Means include timed-out episodes; compare "
            "completion and travel time before drawing energy-efficiency conclusions. "
            "Energy is a simulation-model estimate, not a measurement."
        ),
    }


def load_model(path=DEMO_DIR / "model.zip"):
    """Load only trusted SB3 archives: their metadata may contain pickle objects."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model not found: {path}")
    # Check artifact integrity before SB3 deserializes its trusted metadata.
    metadata = json.loads(path.with_name("metadata.json").read_text(encoding="utf-8"))
    if metadata["schema_version"] != 1:
        raise ValueError("Unsupported example metadata schema")
    if sha256(path) != metadata["model_sha256"]:
        raise ValueError("Model SHA-256 does not match metadata.json")
    return _sb3().load(path, device="cpu"), metadata


def train(output, *, steps=20_000, seed=42, env_id=ENV_IDS[0], learning_starts=500):
    """Train on CPU, save a demo checkpoint, and evaluate in fresh environments."""
    if steps <= 0 or seed < 0 or learning_starts < 0:
        raise ValueError(
            "steps must be positive; seeds and learning_starts non-negative"
        )
    sac = _sb3()
    import torch
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor

    if env_id not in ENV_IDS:
        raise ValueError(f"Unknown environment: {env_id}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    env = None
    try:
        env = Monitor(make_env(env_id))
        check_env(env.unwrapped)
        model = sac(
            "MlpPolicy",
            env,
            device="cpu",
            seed=seed,
            learning_starts=learning_starts,
            verbose=1,
            **copy.deepcopy(SAC_SETTINGS),
        )
        start = time.perf_counter()
        model.learn(total_timesteps=steps)
        duration = time.perf_counter() - start
        path = output / "model.zip"
        model.save(path)
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            revision = None
        metadata = {
            "schema_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "algorithm": "stable_baselines3.SAC",
            "env_id": env_id,
            "environment_kwargs": {},
            "training_seed": seed,
            "training_steps": int(model.num_timesteps),
            "training_seconds": duration,
            "learning_starts": learning_starts,
            "settings": SAC_SETTINGS,
            "device": "cpu",
            "torch_threads": 1,
            "python": platform.python_version(),
            "platform": f"{platform.system()} {platform.machine()}",
            "versions": {
                name: version(name)
                for name in (
                    "gym-longicontrol",
                    "gymnasium",
                    "numpy",
                    "torch",
                    "stable-baselines3",
                )
            },
            "source_base_revision": revision,
            "training_script_sha256": sha256(__file__),
            "model_sha256": sha256(path),
            "purpose": "Getting-started demonstration, not an optimized controller",
        }
        write_json(output / "metadata.json", metadata)
        # Exercise the same loading path that a new user will follow.
        loaded, _ = load_model(path)
        report = compare(loaded, env_id=env_id)
        report["model_sha256"] = metadata["model_sha256"]
        write_json(output / "evaluation.json", report)
        return output
    finally:
        if env is not None:
            env.close()
        torch.set_num_threads(previous_threads)


def plot_rollout(result):
    """Return a Matplotlib figure; works in notebooks and headless test runs."""
    import matplotlib.pyplot as plt

    rows = result["trajectory"]
    elapsed = [row["elapsed_time_s"] for row in rows]
    figure, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7))
    axes[0].plot(elapsed, [row["velocity_km_h"] for row in rows], label="Vehicle")
    axes[0].step(
        elapsed,
        [row["speed_limit_km_h"] for row in rows],
        where="post",
        label="Speed limit",
    )
    axes[0].set_ylabel("Speed [km/h]")
    axes[0].legend()
    axes[1].plot(elapsed, [row["acceleration_m_s2"] for row in rows])
    axes[1].set_ylabel("Acceleration [m/s²]")
    axes[2].plot(elapsed, [1000 * row["total_energy_kwh"] for row in rows])
    axes[2].set_ylabel("Net energy [Wh]")
    axes[2].set_xlabel("Time [s]")
    figure.tight_layout()
    return figure


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train a new CPU demo agent")
    training.add_argument("--output", type=Path, default=Path("runs/sb3-demo"))
    training.add_argument("--steps", type=_positive, default=20_000)
    training.add_argument("--seed", type=_nonnegative, default=42)
    training.add_argument("--env", choices=ENV_IDS, default=ENV_IDS[0])
    training.add_argument("--learning-starts", type=_nonnegative, default=500)
    demo = commands.add_parser("demo", help="Evaluate a trusted, already trained agent")
    demo.add_argument(
        "--model",
        type=Path,
        default=DEMO_DIR / "model.zip",
        help="Trusted SB3 archive with metadata.json; never load unknown files",
    )
    demo.add_argument("--episodes", type=_positive, default=5)
    demo.add_argument("--seed", type=_nonnegative, default=1001)
    demo.add_argument("--report", type=Path, help="Write a new JSON comparison report")
    demo.add_argument(
        "--render", action="store_true", help="Also show the first rollout"
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            output = train(
                args.output,
                steps=args.steps,
                seed=args.seed,
                env_id=args.env,
                learning_starts=args.learning_starts,
            )
            print(f"Saved model, metadata and evaluation to {output}")
        else:
            if args.report is not None and args.report.exists():
                raise FileExistsError(f"Report already exists: {args.report}")
            model, metadata = load_model(args.model)
            env_id = metadata["env_id"]
            report = compare(
                model, env_id=env_id, seeds=range(args.seed, args.seed + args.episodes)
            )
            report["model_sha256"] = metadata["model_sha256"]
            print(json.dumps(report, indent=2, allow_nan=False))
            if args.report is not None:
                write_json(args.report, report)
            if args.render:
                rollout(model, env_id=env_id, seed=args.seed, render=True)
    except (ImportError, OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
