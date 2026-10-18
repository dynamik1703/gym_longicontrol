# Results

## Headline

Numeric seed exclusion is not content exclusion. The exact critical regression found
the expected three Development-2006 collisions, all of which are exact full-task
collisions for fixed Constrained V2. Neither 500-task critical stream collided with any
Validation route/full task.

The implemented generator has 265,949,468 possible route contents, but the distribution
is strongly skewed: 20% of draws fall in a 351-route one-transition stratum. This makes
repeats appreciably more common than the nominal seed namespace suggests.

## Critical results

- T_HIST: 500 tasks, 482 unique routes/full tasks, 3.6% duplicate episode rate,
  maximum multiplicity 3. Development collision at episode 158 / RNG draw 316 with
  seed 2006. No Validation collision.
- T_CURRENT: 500 tasks, 490 unique routes/full tasks, 2.0% duplicate episode rate,
  maximum multiplicity 4. Development collisions at episode 163 / reset seed
  1418256283 and episode 242 / reset seed 274583042, both with seed 2006. No
  Validation collision.
- Empirical Development match rates are 0.002 and 0.004 per episode respectively.
  Empirical Validation match rates are zero in these finite samples; this is not a
  theoretical zero-probability statement.

## Completed studies

Exact reconstructions show clean Validation for current MVE matched Real-Only, H=1
Physics MBRL, HER, CRL, and LLM Reward final. Development is clean for H=1 Physics MBRL
and HER; it has full-task overlap for current MVE, CRL, and LLM Reward.

Historical V2, Requirement-Conditioned RL, matched scalar SAC, and Binary Success are
`PARTIAL` because their artifacts do not preserve enough exact runtime/content
provenance. Their Development and Validation integrity classifications remain
`UNKNOWN`; current-runtime collision counts are presented only as warnings. In
particular, Requirement-Conditioned seed 47 has an indicative Validation-3006 match,
which cannot be promoted to a historical leakage claim without the missing provenance.

No historical numerical result was modified. Development overlap can make repeatedly
observed calibration curves optimistic; Validation integrity carries more weight for
reported final generalization.

## Decision

Adopt Policy R: reserved numeric seeds **or** reserved full-task hashes **or** reserved
route hashes forbid training insertion. A claim about unseen constrained-control tasks
should not train on identical infrastructure under a different deadline.

Repair the stopped 2×2 with Option 1: preserve both exact training streams and pre-freeze
a new route-clean diagnostic set. Eighteen designed candidates and both hashes are in
[`SPLIT_POLICY.md`](SPLIT_POLICY.md); none was evaluated.

## Verification and safeguards

- Fingerprint correctness, 2006 regression, stream reconstruction, duplicate
  regressions, Development/Validation static audit, Paper seal assertions, candidate
  cleanliness, and gate behavior are covered by
  `tests/benchmark/test_content_split_audit.py`.
- Final audit suite: 10 passed. Adjacent stream/provenance/seal subset: 4 passed. Ruff
  lint and format checks passed, deterministic JSON regeneration matched, and
  `git diff --check` passed.
- A broader adjacent invocation reached 65 passed and 2 optional-render skips; 8 tests
  requiring FSRL/Tianshou could not run because pinned FSRL's transitive PyBullet 3.2.7
  failed to build on the available macOS toolchain. The non-PyBullet provenance/seal
  subset was run separately and passed.
- `assert_task_not_reserved` implements S/F/R enforcement and is tested.
- Static audit computation consumed 0 simulator transitions. Required repository
  verification executed 476 test-only state advances in pre-existing generic
  environment/dynamics tests; none used Development, Validation, or Paper tracks.
  Policy-training transitions and policy-evaluation episodes remained 0.
- Validation was not rerun. Only static task definitions were hashed.
- Paper-final `4000..4017` was not instantiated, fingerprinted, compared, or inspected.
  Only numeric rejection membership was verified.
- The machine-readable episode/reset/hash ledger and all study-level collision rows are
  in [`results.json`](results.json).
