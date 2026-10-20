# Hybrid RL Resource Report

This is a bounded preparation benchmark. Synthetic learner updates use
random tensors only; no environment transition trained an RL policy.

- Synthetic update: `0.003008 s/update`.
- Shield: `0.000204 s/action` and `10.138` exact physics calls/action on the fixture.
- Projected three-policy study: `0.554 h` including a 3× uncertainty factor.
- Projected retained disk: `2.208 GB`.
- Gates: runtime `True`, disk `True`.
