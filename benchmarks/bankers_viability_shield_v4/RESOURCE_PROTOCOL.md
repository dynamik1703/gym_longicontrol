# New V4 resource preregistration

Hard limits: 21600 active CPU seconds, 3221225472 retained bytes, 72 Development
pairs (18 tasks per four frozen controllers). Multiplier 1.5 remains unchanged.
No task construction or performance evidence enters resource coefficients.

After synthetic/equivalence/corruption checks, freeze and commit all V4 sources,
tests, canonical.json and these protocols. Then execute nine NEW V4 native
synthetic fixtures and 28 NEW V4 method/speed probes, exactly once. Each probe
uses constant100, five sequential native steps, at initial speeds
[0,20/3.6,10,60/3.6,20,25,25.5] m/s. Each method pays its own maximum over these
same V3 bands. Keep the actual speed distribution. Step0 inherited diagnostics
conservatively charge 20% full-grid diagnostics rather than ~1% in Development.

For method m, c_m=max_band((online wall + full independent physical replay wall
+ canonical compressed write/read wall)/steps). Online includes actual method
control, fsynced compact journal, one mask where applicable, full selected-action
verification and continuous-proposal diagnostic. Record mask time, selected full
verification, complete replay, write/read, compressed and raw journal bytes,
peak RSS and measured source/sealing overhead separately.

Projected CPU = 1.5*18*1800*sum_m(c_m) + measured preflight CPU + measured quality
CPU (tests and differential, charged once) + 900 s source/preparation/reporting.
Measured walls conservatively bound CPU for these sequential workloads. The
quality term differs from V3's disclosed double charge of differential CPU: V4
charges measured work exactly once. The 900 s allowance stays unchanged.

Projected peak storage = current V4 package + inherited V1/V2/V3 evidence and
production source/assets footprint + 1.5*18*1800*sum_m(max_band compressed bytes
per step) + 1800*max_all(raw journal bytes per step) + 1800*max_all(compressed
bytes per step) for one simultaneous atomic publication temporary file + 16 MiB
for panel/manifests/results/reports. This explicitly adds inherited evidence and
the compressed temporary publication compared with V3. Gzip output and live
journal coexist until independently verified result publication. Source/result
manifests, native recovery evidence and measured package peak are accounted.
There is no recovery/restart protocol. One pair runs at a time; there are no
hidden full candidate trajectories. Shared Git object history and unrelated
research packages are not V4 scientific retained artifacts; the referenced
inherited scientific packages and production assets are included conservatively.

Record old V3 and new V4 per-controller maxima and reduction separately; changed
evidence headers/identities are included. Budget checks run during execution;
projection failure stops before Development namespace/task construction. No
sample reduction, margin change, predicate weakening or replay omission is allowed.
Finite probes are a planning bound, not a universal hard real-time proof.
