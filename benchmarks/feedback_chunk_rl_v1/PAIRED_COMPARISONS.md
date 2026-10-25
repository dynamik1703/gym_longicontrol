# Paired Comparisons

The integrity stop prevents every preregistered learned comparison. In
particular, CHUNK3 versus matched RESIDUAL1S, CHUNK3 versus Hybrid, and CHUNK3
versus AdaptivePrice are not estimable and receive no success-level decision.

For transparency, the completed comparator blocks permit paired descriptive
comparisons against AdaptivePrice on 36 jointly feasible routes. Saving is
`100 * (Adaptive energy - comparator energy) / Adaptive energy`; positive
values favor the named comparator. `Delta E` is the signed arithmetic mean of
`Adaptive energy - comparator energy` despite the legacy analysis helper's
`mean_absolute_energy_difference` field name.

| Comparator | Pairs | Mean saving % | Median % | Range % | Mean Delta E kWh | Descriptive 95% route-bootstrap interval for mean % |
|---|---:|---:|---:|---:|---:|---:|
| CAUSAL60 | 36 | -19.875 | -18.202 | [-32.770, -2.304] | -0.033537 | [-21.963, -17.704] |
| TIME_AWARE | 36 | -5.665 | -6.308 | [-14.744, 7.412] | -0.009322 | [-6.822, -4.408] |
| FixedPrice MPC V2 | 36 | 0.107 | -0.214 | [-6.744, 8.302] | 0.000185 | [-0.631, 0.901] |
| Frozen Hybrid H11 | 36 | -4.774 | -5.316 | [-13.365, 6.660] | -0.007859 | [-5.814, -3.653] |
| Frozen Hybrid H29 | 36 | -4.442 | -4.890 | [-13.337, 6.432] | -0.007291 | [-5.606, -3.296] |
| Frozen Hybrid H47 | 36 | -4.048 | -4.618 | [-11.856, 6.229] | -0.006636 | [-5.070, -2.973] |

Intervals are post-stop descriptive percentile intervals from 10,000
route-level bootstrap resamples. Fixed RNG seeds 11200000 through 11200005
were assigned in table order. They were not used for stopping, method
selection, or the final decision. Route count is 36 throughout; repeated
controller or training seeds are not treated as independent routes.

No additional-feasibility-failure claim can be made for CHUNK3 because it has
no Diagnostic outcomes. The preregistered Level B 1% and Level C 3% thresholds
are therefore unevaluable.
