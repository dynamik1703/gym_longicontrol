# LongiControl reward-design task

Control a vehicle along a route.

Desired operational behavior, in priority order:

1. complete the route;
2. complete it within 140 seconds;
3. respect the speed limit throughout the route;
4. among successful behavior, prefer lower net energy consumption.

The training signal must be a deterministic scalar reward computed after each
simulator transition. Physical task success is measured separately from the
reward and must not be redefined by the reward designer.

Design the reward only from the fields documented in `ALLOWED_SIGNALS.md` and
the API in `REWARD_API.md`. Do not assume access to track identity, external
scores, saved files, network services, environment variables, optimal actions,
precomputed trajectories or other experiment artifacts.

