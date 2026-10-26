# Provisional inherited reward/cost contract

The component retains corrected Hybrid's physical native signals:

- `r = -1000 * signed step energy [kWh]` (regeneration preserves its sign).
- `c_speed = max(0, endpoint speed - endpoint physical limit) [m/s]`.
- `c_terminal = 1` at a true episode end if incomplete or elapsed time >140 s;
  otherwise zero. Environment TimeLimit at 1800 native steps is a true task end.
- `c_intervention = abs(native proposal - native execution)`, with the inherited
  fixed actor regularizer 0.05. This is not a third hard constraint.

Route progress, completion, deadline, energy and speed remain distinct recorded
quantities. No progress reward or completion bonus was added. Standing still
eventually incurs terminal task failure under the inherited constraint; energy
alone does not establish success. This contract cannot guarantee that a short
pilot learns meaningful progress, and no such result was observed here.

Training-budget cutoff alone is not terminal failure. Deadline remains the
policy's responsibility. Exact external feasibility requires completed route,
travel time <=140 s and zero positive speed excess; failed tasks cannot be
rescued by low or negative energy. V2's route-dependent dense deadline signal
was rejected as a causal-boundary violation rather than silently rewritten.
No weights were tuned and no scientific reward freeze was released.
