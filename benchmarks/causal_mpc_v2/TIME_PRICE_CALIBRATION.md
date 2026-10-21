# Time-Price Calibration

The reference logarithmic range `1e-5 ... 1e-2 kWh/s` is retained. It contains
the V1 observed marginal transition prices (`0.001625 ... 0.007257 kWh/s`) and
keeps units explicit.

The fixed selector grid is normalized `[-1, -0.5, 0, 0.5, 1]`. The adaptive
grid uses `(high reserve, low reserve)` pairs `(12,2)`, `(18,2)`, `(24,2)`
seconds with the same monotone mapping as V1. All candidates are declared here
before Calibration. The hierarchy is maximum hard feasibility, minimum mean
feasible energy, then minimum measured planner time and deterministic parameter
order. No Capability result may alter this mapping or selection.

