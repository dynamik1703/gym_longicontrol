# Fixed controller comparison
| Arm | Proposal | Executed action | Safety interpretation |
|---|---|---|---|
| CAUSAL60 | frozen causal continuous law | exact proposal | original physical reference |
| LEGACY_NATIVE_POST_GREEDY | +1 | +1 if inherited predicates accept, else frozen causal replacement | native proposal variant, not original capped controller |
| BANKERS_GREEDY | +1 preference | largest certified grid point | conditional speed certificate, no deadline guarantee |
| BANKERS_MASKED_CAUSAL60 | frozen causal continuous law | nearest certified grid point, ties smaller | grid approximation, not bit-identical CAUSAL60 |

All arms have identical original tasks and causal information. The A shadow mask
has no influence. Matched-state legacy/Banker certificate comparisons are distinct
from B/C performance differences, which also reflect proposal/action selection.
Full records retain continuous proposal, executed grid action, exact successors,
all legacy flags and all 21 Banker certificates, permitting action-difference,
trajectory and disagreement analysis. A failed inner certificate proves only lack
of certification. Deadline-only rejections do not falsify speed invariance.

Development distributions and task-paired conclusions are reported only after
the preflight gates pass and all scheduled results have been verified and sealed.
