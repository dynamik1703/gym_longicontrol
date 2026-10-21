# Calibration Results

Panel: `MPC_V2_CALIBRATION` (`24a5bdfdab2d8cf2afa4083e977ef49a24b5f1d2b35965c68a002d7a70e76500`).
Budget consumed exactly: 96 episodes.

## Fixed-price candidates

| Action | Lambda (kWh/s) | RSR | Mean energy (kWh) | Planner time (s) |
| ---: | ---: | ---: | ---: | ---: |
| -1.00 | 0.00001000 | 12/12 | 0.173653 | 193.219 |
| -0.50 | 0.00005623 | 12/12 | 0.173667 | 193.263 |
| 0.00 | 0.00031623 | 12/12 | 0.173205 | 186.333 |
| 0.50 | 0.00177828 | 12/12 | 0.176000 | 180.263 |
| 1.00 | 0.01000000 | 12/12 | 0.183296 | 176.115 |

## Adaptive candidates

| High/low reserve (s) | RSR | Mean energy (kWh) | Influence | Counterfactual first-action difference |
| --- | ---: | ---: | ---: | ---: |
| 12/2 | 12/12 | 0.176086 | 35.861% | 47.593% |
| 18/2 | 12/12 | 0.175678 | 34.579% | 46.773% |
| 24/2 | 12/12 | 0.175363 | 34.426% | 47.071% |

## Frozen selection

Fixed action `0.0` (lambda `0.00031622776601683783` kWh/s); adaptive reserves `24.0/2.0` s. No Capability outcome was observed before this freeze.
