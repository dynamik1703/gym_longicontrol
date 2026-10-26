# Resource protocol frozen before preflight or Development
Hard gates: <=21600 active CPU seconds, <=3221225472 retained bytes, exactly
18 Development tasks per controller if all preconditions pass; zero RL updates.

Synthetic distribution: constant100 road, seven initial speeds
[0,20/3.6,10,60/3.6,20,25,25.5] m/s, five sequential native decisions per method
and speed. Each of four methods is measured separately, including its real
control law, durable per-step journal, masks where applicable, full selected
production certification and masked-CAUSAL60 continuous-proposal diagnostic, complete independent production replay, canonical gzip
write/readback. Step0 diagnostic makes these five-step probes conservatively
charge 20% full-grid diagnostics versus the actual deterministic ~1% schedule.
Declare all measured speeds and actual successor speed distributions.

For each method m, c_m is the maximum over its seven speed probes of
(online wall + independent replay wall + evidence I/O wall)/native steps.
Frozen projected active seconds = 1.5 * 18 * 1800 * sum_m(c_m)
+ measured preflight CPU + measured tests/differential/baseline CPU + 900 seconds
preparation/source-check/reporting allowance. Wall times conservatively bound
charged CPU. Supplemental and matched latency CPU are included in quality phase
accounting. No Development outcome enters any coefficient or formula. The maximum
per-method probe cost is charged at EVERY possible native step, regardless of
future performance. Worst cross-method cost and speed-band distribution are
reported separately; no finite benchmark establishes a hard real-time bound.

Storage projection = existing retained package bytes + 1.5*18*1800 * sum_m(max
compressed bytes/native step for m) + largest measured raw bytes/step *1800
for one live journal +16MiB reporting allowance. Peak RAM is measured separately.
An actual budget monitor runs during execution; projection failure stops before
constructing the Development split. No panel reduction or verification omission.

Prospective RL assumes 300000 transitions per seed11/29/47 as a scenario, NOT a
new training authorization. Optimistic/conservative measured mask, full final,
replay and evidence costs are reported; neural updates/checkpoints are not
measured here, so total training feasibility cannot be established or claimed
from mask-only timing, even when a component projection fits24h.
