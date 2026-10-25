# Resource report

Before scientific authorization, representative fixed-fixture evaluations run
in both native runtimes. The conservative projection includes a 2× uncertainty
factor and write/verification overhead. Hard gates are 12 active compute hours
and 4 GiB retained V2 artifacts. Machine-readable measurements are written to
`resource_measurements.json`.

Eight representative rollouts took 40.169 s in Worker C and 45.558 s in Worker
H. With the frozen 2× uncertainty factor and 10% write/verification allowance,
the projected active evaluation time is 5,863.35 s (1.63 h), below 43,200 s.
Projected retained data is 85,659,680 bytes, below 4,294,967,296 bytes. Both
hard gates pass.
