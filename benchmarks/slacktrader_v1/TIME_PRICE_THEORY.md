# Time-Price Theory and Calibration

The physical objective is

`J = E_horizon + lambda * T_horizon + V_terminal`,

where energy is kWh, time is seconds and `lambda` is kWh/s. A larger price
makes elapsed time more expensive. For two candidate trajectories, their
finite-difference marginal price is approximately

`(E_fast - E_slow) / (T_slow - T_fast)`.

The action map is logarithmic:

`lambda(a) = 10^-5 * (10^-2 / 10^-5)^((a + 1)/2)`, `a in [-1, 1]`.

The Development counterfactuals observed transition prices from
`0.001625` to `0.007257` kWh/s, inside the frozen range. The deliberately wider
lower tail permits energy-priority behavior; the upper endpoint crosses every
observed transition. Bounds and mapping are now frozen and must not be changed
after any learned-policy run.

For identical causal states and candidate sets, low/medium/high prices produced
two distinct selected durations and native first actions on all four capability
tracks. Duration was non-increasing with price in every test. The interface
therefore passes the non-degeneracy gate, even though many ordinary states have
a single physically dominant candidate and the end-to-end planner remained
too fallback-like for the study's scientific gate.
