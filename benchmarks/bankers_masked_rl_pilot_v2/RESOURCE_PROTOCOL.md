# Prospective complete resource gate (before measurement)

Limits: 8 hours total active compute, 2 GiB retained new study artifacts, 150000
real native transitions and 15000 optimizer updates. The three-arm matrix cannot
be shrunk and no required certificate/audit may be dropped to pass.

Measure each arm separately on explicit unreserved constant-100 synthetic physics
fixtures at speeds 0,20/3.6,10,60/3.6,20,25,25.5 m/s, five steps per speed. Record
wall and process CPU. Include inference, environment, current and successor mask,
original final certification, replay, independent audit, durable intent/journal,
compression, all optimizers, full-state checkpoint save/load, recovery, Diagnostic
controllers/native evidence/complete independent verification and final hashing.
No held-out numbered task or policy-performance outcome is used.

Fixed conservative time model: for each arm use the maximum nonsampled complete
native-step time across bands x50000, plus maximum nonsampled disk audit/read time
x50000, plus 500 sampled-grid extras in each audit. Add 5000 x maximum measured
full-batch optimizer+sampling time, 52 checkpoint writes, and one recovery reserve
(load+50000 journal reads+1000 native/update reconstructions) per arm. Diagnostic
uses six separate maximum end-to-end five-step rates x12x1800, including complete
disk verification. Add 250 measured source-check equivalents (including pair gates and reserve), three full artifact hash passes,
and a 600s final analysis/report/sealing allowance. Multiply complete future work
by 2. Add measured/prior engineering/probe cost and one complete latest-engineering-suite
wall-time reserve for the prospective sealed preflight. Wall-rate estimates conservatively
upper-bound active CPU and include I/O; actual CPU and wall are reported separately.

Storage: all per-arm native budgets x maximum observed compressed bytes/step,
all Diagnostic budgets x their maxima, the largest live journal, five uncompressed
checkpoint byte bounds (three finals, one technical, one atomic temporary), plus
16MiB metadata. Multiply by 2. Include all scientific outputs and publication peaks.
Temporary deterministic test fixtures may be removed after retaining XML/source
identities. Reused V1 dependency environments are not new scientific evidence.

If either complete bound fails, V2_RESOURCE_STOP before scientific training.
No matrix adjustment or selective omitted cost after seeing the projection.
Later one-method/three-method x3 seeds x300k models keep 30k updates/run and a
single final 12-task panel and are prospective only; no automatic <=24h claim.
