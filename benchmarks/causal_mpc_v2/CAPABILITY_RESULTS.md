# Capability Results

Panel: `MPC_V2_CAPABILITY` (`0744c1962d7f85c4625183ab247ebc2c97b15c4c5ed1439ce4bb46eb0c6520c8`).
Exactly 72 controller-task pairs were executed (18 per controller).

| Controller | RSR | Completion | Deadline | Speed | Mean/median feasible energy (kWh) | Mean time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CAUSAL60 | 18/18 | 18/18 | 18/18 | 18/18 | 0.196323/0.191102 | 92.417 |
| TIME_AWARE | 18/18 | 18/18 | 18/18 | 18/18 | 0.180326/0.174067 | 128.811 |
| FIXED_PRICE_V2 | 18/18 | 18/18 | 18/18 | 18/18 | 0.172002/0.167216 | 101.922 |
| ADAPTIVE_PRICE_V2 | 18/18 | 18/18 | 18/18 | 18/18 | 0.170666/0.166386 | 105.461 |

## Paired energy

Adaptive versus TIME_AWARE: 5.422%.
Adaptive versus FixedPrice V2: 0.801%.
FixedPrice V2 versus TIME_AWARE: 4.644%.

## Executed authority and Shield

FixedPrice: 5208/18346 influenced native actions, 387 overrides, 12751 fallback-identical proposals and 40.132% counterfactual first-action differences.
AdaptivePrice: 6274/18983 influenced native actions, 159 overrides, 12550 fallback-identical proposals and 44.848% counterfactual first-action differences.
Distinct executed native-action hashes were 17/18 and 17/18. Mean selected Adaptive terminal-energy contribution per solve was 0.044400 kWh; this is a receding-horizon diagnostic, not episode energy.

## Mechanism interpretation

Fixed V2's 4.644% saving versus TIME_AWARE is the combined effect of the richer profiles and corrected terminal heuristic; those two changes were not separately identified. Adaptive's 0.801% over Fixed isolates the state-dependent price rule and missed its 1% gate. Shield interventions quantify blocked proposals but do not identify an energy counterfactual. No post-Capability ablation was performed.

The privileged reference was preregistered as omitted, so no new
privileged-reference gap is claimed.

Scientific classification: `MPC_IMPROVED_BUT_RL_VALUE_UNCLEAR`.
