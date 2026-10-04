# Stable-Baselines3 scalar baseline

This study asks whether the instability observed with the historical SAC
implementation persists with current Stable-Baselines3 SAC and PPO. It freezes
the V2-B Balanced reward and the 140-second physical task before training.

The canonical protocol is in `canonical.json`. Both algorithms use their SB3
defaults, including their default policy networks. Observations already have a
declared `[0, 1]` range, so `VecNormalize` is not used. Rewards are not normalized
or transformed. A callback evaluates deterministic policies every 50,000
environment interactions and stops PPO exactly at the same 300,000-step sample
budget as SAC, even if that interrupts PPO's last rollout before an update.

See `PROTOCOL.md` for the frozen scientific protocol and
`CUSTOM_SAC_DIAGNOSTIC.md` for the bounded implementation comparison.

Run artifacts (models, raw checkpoint evaluations, and diagnostic logs) belong
under the ignored `runs/` tree:

```text
runs/scalar-sb3-YYYYMMDD/
  sac/training-seed-11/step-050000/
  ppo/training-seed-11/step-050000/
```

Run the complete preregistered study with:

```bash
python -m benchmarks.scalar_sb3.experiment \
  --output-dir runs/scalar-sb3-YYYYMMDD \
  --workers 2 \
  --device cpu
```

Seeds 4000–4017 are configuration-only reserved identifiers. The experiment
runner never evaluates them.
