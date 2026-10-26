# Development results: not opened

BANKERS_DEV_V1 was not instantiated because the original preflight integrity
gate failed. Zero selected/generated Development tasks, zero controller-task
performance episodes, no A/B/C RSR estimate and no feasible-only energy result.
All such machine-readable values are null or explicitly NOT_OPENED, not zero
performance scores. The 18-task protocol remains a proposal, not a completed
study. Historical Validation and Paper remain untouched.

## Retained synthetic scope

The constant-20 correctness fixture executed 1800 steps, then truncated at
179.999999999994 s with final position
990.625158849 m. Completion=false; deadline
failed; exact speed excess=0 at every retained endpoint; signed physical energy
0.169454127688 kWh. This energy is not a feasible-only estimate
because the original task requirements failed. A constant 20 km/h route cannot
satisfy the 140 s requirement; conservative braking may likewise preserve speed
while sacrificing progress. Historical metadata independently prove task
10200029 infeasible under the frozen deadline (inherited FEASIBILITY_PROOF.md);
no new episode or classification for that identity was created.

Every retained state had a certified -1 backup. Certified native action count:
11 choices at 899 steps; 12 at 879; 13 at one; 21 at 21. The corresponding
position/velocity successors were physically distinct. Mean count
11.606111; admitted fraction
0.552672; meaningful successor spread >=0.05 m/s at
100.00% of these states. Positive acceleration
was available at 50.06%.
The full velocity-spread distribution and rejected-positive-action fraction
are retained in retained_fixture_verification.json. These are synthetic
low-speed measurements, insufficient to decide usefulness across the required
Development distribution.

Matched inherited-gate comparison on the same 37800 synthetic native proposals:
`{'BANKERS_ONLY': 20870, 'BANKERS_ONLY_DEADLINE': 20870, 'BOTH_ACCEPT': 21, 'BOTH_REJECT': 16909}`. Most new-only admissions reflect the separate
inherited deadline diagnostic, not proof that the inherited speed shield is
incorrect. No Development controller comparison was executed. The old
controller and historical classifications remain unchanged.

The offline search used 320 synthetic states and 6720 proposed grid actions.
6470 admitted witnesses and successor backup certificates passed independent
production-dynamics checks, with no core invariant counterexample. These are
physics predictions/correctness probes, not additional performance or training
episodes. Counterexample_search.json retains the separate failed latency
shortcut example, outside the old capped proposal domain.

All 1800 retained native records, all 37800 mask classifications, selected
certificate digests, actual right-endpoint limits, signed energy, lifecycle,
causal observations/memory and every applicable completion bound were replayed
from disk. The original readback integrity failure is still a failed gate.
