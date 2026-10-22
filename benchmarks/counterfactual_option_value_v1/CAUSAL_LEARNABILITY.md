# Causal Learnability

Frozen family: `linear`; energy margin: `0.2 Wh`; risk threshold: `0.05`.

- Energy-difference MAE: 0.0008868412673094417 kWh
- Action-ranking accuracy: 0.3020833333333333
- Meaningful-sign accuracy (|difference| >= 0.2 Wh): 0.6304347826086957
- Always-reference regret: 0.051644238292194417 kWh
- Frozen-predictor regret: 0.026635335774641083 kWh
- Heuristic-runner-up regret: 0.021662671648134763 kWh
- False-safe decisions: 0
- Harmful predicted improvements: 6
- Abstention rate: 0.6354166666666667
- Routes with useful predictions: 6

Route-bootstrap uncertainty and every selected decision are retained in
`test_evaluation.json`. No Test-dependent tuning followed scoring.
