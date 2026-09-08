"""Run one seeded random-policy rollout from a repository checkout."""

import argparse

import gymnasium as gym


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    env = gym.make(
        "gym_longicontrol:StochasticTrack-v1",
        render_mode="human" if args.render else None,
    )
    try:
        env.reset(seed=args.seed)
        env.action_space.seed(args.seed)
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            done = terminated or truncated
        print(
            f"Distance: {info['position_m']:.1f} m, "
            f"energy: {info['total_energy_kwh']:.4f} kWh"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
