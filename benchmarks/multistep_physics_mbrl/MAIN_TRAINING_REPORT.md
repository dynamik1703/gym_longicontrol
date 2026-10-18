# Main-training report: Multi-Step Physics MVE

## Outcome

The frozen six-policy main-training matrix completed in its preregistered serial order.
Every policy reached exactly 300,000 real native transitions and 30,000 RL updates,
with zero synthetic replay rows. The fixed 300k policy is the scientific endpoint in
every case; no best-checkpoint selection was used.

The Development-only outcome is negative. At 300k, matched real-only achieves pooled
RSR 13/27, while Physics MVE achieves 0/27. Physics MVE also falls from 16/27 to 9/27
completion, 13/27 to 9/27 deadline compliance, and 26/27 to 15/27 speed compliance.
Without the stability override this is **D — degradation**. The preregistered overall
classification is **E — unstable**, because Physics MVE seed 47 first exceeds the
1,000,000 critic-loss threshold at 200k and ends above both the critic-loss and target-
magnitude thresholds.

This report is Development-only. Validation 3000–3008 and paper-final 4000–4017 were
not evaluated.

## Final policy states and provenance

| Order | Policy | State | Attempt ID | Resume | Final SHA-256 |
|---:|---|---|---|---|---|
| 1 | matched real-only, seed 11 | COMPLETED | `4daa201b-8f3b-4096-a56d-4182326a41f6` | no | `994568f614c9d2428b805fafafc9526717fa2d6e4ca73783adfcb361feb4090d` |
| 2 | matched real-only, seed 29 | COMPLETED | `8e6415eb-e087-4089-85d4-15c3a1303636` | no | `f32757afc979166793ba2f3725df7511af728dd446de1b2ded4d482a3034f5e8` |
| 3 | matched real-only, seed 47 | COMPLETED | `d0dff0dd-9f8b-4332-97f5-fcba552de63f` | no | `e0262a618f0435f85b792286cdbd9aebb3a6958e01a88ae101724ed8c98cade6` |
| 4 | Physics MVE, seed 11 | COMPLETED | `ed40b66b-bbe2-495a-b882-d40b418ddbc0` | no | `8125ed0cc385e0eead5435e82f9d35b2a69607b1ce54aad8f7b30cc9ea566d9b` |
| 5 | Physics MVE, seed 29 | COMPLETED | `bc990866-050c-49ba-8213-23699ea36ea1` | no | `66ca2345b311ad19f7455bf742316866a00cfdbd52f60bedfdf021fe77b767dd` |
| 6 | Physics MVE, seed 47 | COMPLETED | `e38a7d14-4c57-410b-a5a9-1e0d55205be2` | no | `e1c02028c368111e2807ad75733112c0502d5118aa545795c96ba37ac21aa88e` |

Each final SHA was recomputed from the checkpoint bytes and matched both its metadata
and the run manifest. All final checkpoints contain configuration SHA-256
`d73b05bdffa725b0a0c5a7d831bc926328ec8eb6629d2147fa9fd6dfe709a42a`
and the same frozen source hashes. The execution-authorization commit was `7dc95d9`, a
direct descendant of prepared commit `2115ce37bbdc244efc8a58036e34cb72fc675b52`.

## Counters and PID isolation

| Policy | Real transitions | RL updates | Synthetic replay | Model-target transitions | Real-episode PID updates |
|---|---:|---:|---:|---:|---:|
| real-only 11 | 300,000 | 30,000 | 0 | 0 | 222 |
| real-only 29 | 300,000 | 30,000 | 0 | 0 | 230 |
| real-only 47 | 300,000 | 30,000 | 0 | 0 | 240 |
| MVE 11 | 300,000 | 30,000 | 0 | 451,371,345 | 203 |
| MVE 29 | 300,000 | 30,000 | 0 | 450,517,507 | 240 |
| MVE 47 | 300,000 | 30,000 | 0 | 452,180,488 | 173 |

The saved `pid_real_episode_updates` equals `previous_completed_episodes` in every
final checkpoint. Model target transitions affected critic targets only; synthetic
replay remained zero and no modeled transition entered PID/Lagrange updates.

## Development curves

Entries are counts out of the nine fixed Development tracks at 50k, 100k, 150k,
200k, 250k, and 300k respectively.

| Policy | RSR | Completion | Deadline | Speed |
|---|---|---|---|---|
| real-only 11 | 0, 3, 2, 3, 0, 2 | 0, 9, 9, 9, 9, 4 | 0, 6, 6, 6, 9, 2 | 9, 4, 3, 4, 0, 9 |
| real-only 29 | 0, 0, 5, 6, 4, 5 | 0, 9, 9, 9, 5, 5 | 0, 9, 6, 9, 4, 5 | 9, 0, 7, 6, 9, 9 |
| real-only 47 | 0, 0, 4, 7, 3, 6 | 0, 9, 9, 9, 9, 7 | 0, 9, 8, 7, 8, 6 | 9, 0, 4, 9, 4, 8 |
| MVE 11 | 0, 0, 1, 0, 4, 0 | 0, 1, 3, 9, 6, 0 | 0, 0, 1, 9, 6, 0 | 9, 7, 7, 0, 5, 6 |
| MVE 29 | 0, 0, 0, 0, 0, 0 | 0, 0, 1, 0, 0, 9 | 0, 0, 0, 0, 0, 9 | 9, 9, 9, 9, 9, 0 |
| MVE 47 | 0, 0, 0, 0, 0, 0 | 0, 0, 0, 0, 0, 0 | 0, 0, 0, 0, 0, 0 | 9, 9, 9, 9, 9, 9 |
| pooled real-only | 0, 3, 11, 16, 7, 13 | 0, 27, 27, 27, 23, 16 | 0, 24, 20, 22, 21, 13 | 27, 4, 14, 19, 13, 26 |
| pooled MVE | 0, 0, 1, 0, 4, 0 | 0, 1, 4, 9, 6, 9 | 0, 0, 1, 9, 6, 9 | 27, 25, 25, 18, 23, 15 |

Feasible-only energy is recorded in `main_training_results.json`. It is undefined at
the 300k MVE endpoint because MVE has no feasible final Development episode.

## Final exclusive failure modes

| Policy | Exclusive 300k modes |
|---|---|
| real-only 11 | 2 fully feasible; 2 completed too slowly; 5 partial/crawling |
| real-only 29 | 5 fully feasible; 4 partial/crawling |
| real-only 47 | 6 fully feasible; 1 completed too slowly; 2 partial/crawling |
| MVE 11 | 9 partial/crawling |
| MVE 29 | 9 completed with speed violation |
| MVE 47 | 9 standstill |

Thus the pooled baseline ends with 13 fully feasible, 3 completed too slowly, and 11
partial/crawling episodes. MVE ends with three seed-specific failure regimes: 9
partial/crawling, 9 speed-violating completions, and 9 standstills.

## Target decomposition at 300k

The table reports count-weighted means as `target = real two-step + model expansion +
terminal/bootstrap`; `fraction` is the mean absolute model-expansion fraction of
target magnitude. Real-only uses the untouched FSRL two-step update, so the MVE target
accumulator is intentionally not populated there.

| Seed / critic | Target | Real prefix | Model expansion | Terminal/bootstrap | Fraction |
|---|---:|---:|---:|---:|---:|
| MVE 11 / objective | -1.9235 | -0.0018 | -0.9508 | -0.9709 | 2.0057 |
| MVE 11 / speed | -1.1530 | 0.0514 | -0.2963 | -0.9082 | 2.1586 |
| MVE 11 / deadline | -0.8109 | 0.0243 | -0.3850 | -0.4502 | 2.3593 |
| MVE 29 / objective | -157.5431 | -0.0012 | -66.6299 | -90.9120 | 0.4653 |
| MVE 29 / speed | -138.6998 | 0.1339 | -62.9347 | -75.8990 | 0.5252 |
| MVE 29 / deadline | -155.9755 | 0.0347 | -65.7801 | -90.2301 | 0.5121 |
| MVE 47 / objective | -2,030,932.2586 | -0.0002 | -927,564.4380 | -1,103,367.8205 | 0.4564 |
| MVE 47 / speed | -2,030,461.6538 | 0.0000 | -927,564.4341 | -1,102,897.2197 | 0.4564 |
| MVE 47 / deadline | -2,031,547.8630 | 0.0495 | -927,563.3306 | -1,103,984.5819 | 0.4562 |

Final mean modeled rollout lengths are 58.7463, 58.6761, and 58.9453 steps for MVE
seeds 11, 29, and 47. The requested maximum is 60 and true terminals shorten some
rollouts.

## Anticipatory-speed diagnostic

The number of nine fixed probes with any nonzero predicted speed cost is:

| Policy | 50k | 100k | 150k | 200k | 250k | 300k |
|---|---:|---:|---:|---:|---:|---:|
| real-only 11 | 0 | 0 | 0 | 0 | 1 | 0 |
| real-only 29 | 0 | 0 | 0 | 0 | 0 | 0 |
| real-only 47 | 0 | 0 | 0 | 0 | 0 | 0 |
| MVE 11 | 0 | 0 | 0 | 4 | 0 | 0 |
| MVE 29 | 1 | 1 | 1 | 1 | 1 | 5 |
| MVE 47 | 2 | 2 | 2 | 2 | 2 | 2 |

All records have `controls_actions = false`. These probes never planned, selected an
action, altered evaluation, or selected a checkpoint. Nonzero MVE sensitivity therefore
did not translate into robust compliant Development behavior.

## Lambda, alpha, and loss stability

| Policy | Max lambda | Final `(lambda_speed, lambda_deadline)` | Final alpha | Max abs critic loss | Max abs target | Gate |
|---|---:|---|---:|---:|---:|---|
| real-only 11 | 9.1692 | (2.1414, 1.6620) | 0.00453 | 0.5832 | n/a | pass |
| real-only 29 | 4.9837 | (1.4081, 1.0868) | 0.00099 | 0.1560 | n/a | pass |
| real-only 47 | 11.2563 | (1.3508, 0.9846) | 0.00046 | 0.2703 | n/a | pass |
| MVE 11 | 35.4720 | (2.1999, 2.0073) | 0.06101 | 67.4656 | 222.9799 | pass |
| MVE 29 | 23.2511 | (21.2218, 2.5339) | 0.62727 | 6,090.5747 | 650.1675 | pass |
| MVE 47 | 9.1176 | (1.8749, 3.6892) | 7,056.3613 | 2,242,828,837.1778 | 4,272,062.7342 | **fail** |

All monitored values remained finite and no multiplier reached 1,000. Seed 47's
critic-loss series at 50k intervals is 34,085.56; 146,214.74; 510,087.25;
5,657,317.97; 105,020,606.92; and 2,242,828,837.18. The first preregistered breach is
therefore the 200k critic-loss record. Its target magnitude remains below one million at
250k (946,298.88) and reaches 4,272,062.73 at 300k. Alpha also rises monotonically to
7,056.36, although alpha itself was not a separately preregistered threshold.

## Late collapse

Only MVE seed 11 meets the preregistered late-collapse rule: peak Development RSR 4/9
at 250k falls to 0/9 at 300k, a drop of four. Every real-only seed has a peak-to-final
drop of one, while MVE seeds 29 and 47 stay at zero throughout.

## Actual resources

Wall time includes multiple host-sleep intervals and contention. CPU deltas distinguish
active compute where the process sample was retained.

| Policy | Wall time | Process CPU | Checkpoint writes | Peak RSS |
|---|---:|---:|---:|---:|
| real-only 11 | 00:55:05.9 | see aggregate | 00:06:43.4 | 1,106,000 KB |
| real-only 29 | 00:43:52.4 | see aggregate | 00:05:09.1 | 1,620,624 KB |
| real-only 47 | 00:35:24.1 | see aggregate | 00:04:42.3 | 1,760,016 KB |
| MVE 11 | 22:48:06.7 | 10:27:01.9 | 00:04:46.2 | 1,603,376 KB |
| MVE 29 | 73:11:17.8 | 11:28:04.0 | 00:05:12.9 | 1,682,544 KB |
| MVE 47 | 24:27:28.2 | >10:49:57.2 | 00:05:11.8 | 1,564,304 KB |

The exact combined CPU time for the three real-only policies is 01:32:48.7. Per-policy
real-only CPU boundaries were not retained. MVE 47's CPU value is an exact lower bound
through 290k; the worker exited before the final `ps` sample, so the last 10k CPU delta
is intentionally not invented. Total wall span is 122:41:17.1 and observed process CPU
is greater than 34:17:51.8. Checkpoint writes total 00:31:45.6. Maximum observed RSS is
1,760,016 KB. The observed run-directory peak was at least 11,054,420 KB at MVE seed
47's 290k checkpoint.

## Recovery and checkpoint integrity

Each policy has one immutable Attempt and no resume. Thirty verified checkpoint
metadata records and all six 50k scientific checkpoint payloads remain for every
policy. After completion and hash verification, 24 redundant non-scientific recovery
payloads per policy were removed to relieve disk pressure; their metadata remains.
Those removed payload files are not locally recoverable. The retained run directory is
4,989,248 KB, and the cleanup increased free disk space from about 12.6 GB to 18.8 GB.

The launchd matrix process used PID 52983 throughout, returned exit code 0, emitted
`MATRIX_COMPLETE`, and was unloaded after completion. The benign Tianshou single-
environment wrapper warning was the only stderr output observed.

## Reserved splits and stopping point

Validation remains `SEALED` with `opened_at = null` and authorization false. Paper test
remains `SEALED` with `opened_at = null` and authorization false. No Validation or
paper-final actor evaluation was performed or authorized automatically.

**Main training complete.**

**Validation remains sealed and unauthorized.**

**Paper test remains sealed and unauthorized.**
