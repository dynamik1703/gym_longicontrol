# LLM reward search: frozen-winner final results

## Scope and provenance

Phase 3 trained the mechanically selected and immutable winner `g1-c02` from
scratch with Stable-Baselines3 SAC. The reward source SHA-256 is
`714ed4124e98b2e12d1d89a72edb78d9ab1f04de8fd88e2e18bb74350417fd47`.
The frozen protocol is `llm-reward-search-v1`, SHA-256
`7f1da03f98c8d1bd7f3607710ca62bab82f19f86670cd8b27d5458d83f28ca4e`.
The search stayed `CLOSED`; no reward was generated, edited, repaired,
reselected, or tuned during Phase 3.

The final runner was invoked exactly once. Each seed received 300,000 native
simulator transitions and a fresh policy and replay buffer. Training used the
frozen 256x256 SB3-SAC configuration with no observation or reward
normalization. `StochasticTrack-v1` sampled a stochastic track at each training
reset from the seeded environment RNG. Development tracks 2000--2008 were not
a finite training schedule. Only after training, the final policy was evaluated
deterministically on Validation tracks 3000--3008 in a separate environment;
those interactions never entered replay.

The runtime was Python 3.11.13, gym_longicontrol 1.0.0,
Stable-Baselines3 2.9.0, PyTorch 2.14.0, NumPy 2.4.6 and Gymnasium 1.3.0.
Device selection was `auto`, resolving to CPU.

## Primary results

| Training seed | Screening role | RSR | Completed | Completed by 140 s | Speed-compliant, all episodes | Conditional speed compliance | Exclusive failures | Feasible track / energy |
|---:|---|---:|---:|---:|---:|---:|---|---|
| 11 | fixed screening seed | 1/9 (11.1%) | 8/9 | 8/9 | 2/9 | 2/9 | feasible 1; speed 7; incomplete+time 1 | 3008 / 0.197414 kWh |
| 29 | unseen in screening | 0/9 | 0/9 | 0/9 | 9/9 | 1/1 | incomplete+time 9 | none / undefined |
| 47 | unseen in screening | 0/9 | 0/9 | 0/9 | 9/9 | 1/1 | incomplete+time 9 | none / undefined |
| **Overall** | three policy replicates | **1/27 (3.7%)** | **8/27 (29.6%)** | **8/27 (29.6%)** | **20/27 (74.1%)** | **4/11 (36.4%)** | **feasible 1; speed 7; incomplete+time 19** | **one episode / 0.197414 kWh** |

The conditional speed denominator is exactly the screening definition:
completed episodes or incomplete episodes reaching at least 50% of the
1,000 m route. It is shown separately from unconditional compliance. Seeds 29
and 47 are unconditionally speed-compliant mostly because they do not finish;
their low energy therefore is not efficiency.

Travel-time and violation distributions are:

| Seed | Travel time, mean / median / range (s) | Maximum speed excess, mean / median / max (m/s) | Integrated excess, mean / median / max (m) |
|---:|---:|---:|---:|
| 11 | 109.878 / 108.500 / 75.000--180.000 | 0.697 / 0.730 / 1.719 | 4.411 / 4.450 / 11.222 |
| 29 | 180.000 / 180.000 / 180.000--180.000 | 0 / 0 / 0 | 0 / 0 / 0 |
| 47 | 180.000 / 180.000 / 180.000--180.000 | 0 / 0 / 0 | 0 / 0 / 0 |
| **Overall** | **156.626 / 180.000 / 75.000--180.000** | **0.232 / 0 / 1.719** | **1.470 / 0 / 11.222** |

Only the single seed-11/track-3008 episode was feasible, so its mean and median
feasible energy are both 0.197414 kWh. Feasible energy is `null`, not zero, for
seeds 29 and 47.

## Completion and resource accounting

| Seed | Model timesteps | Gradient updates | Training wall time (s) | Validation interactions | Model SHA-256 |
|---:|---:|---:|---:|---:|---|
| 11 | 300,000 | 299,900 | 2,538.810 | 9,889 | `a4e61c04634c41ea493cef98698e08fed356bdbecb3a286f2f92dbc5768fc608` |
| 29 | 300,000 | 299,900 | 2,020.819 | 16,200 | `faefdcd99abddb112a88f6b58b25d61b473ee077fdcc0e4298135baef305221d` |
| 47 | 300,000 | 299,900 | 1,982.399 | 16,200 | `4a0cd115642628bd6e356890b4ba61dd7a648c3fb7a1ef48fa228d4f8473e3ce` |

Saved SB3 models were loaded after the run and each reported exactly 300,000
timesteps. The fixed search consumed 550,000 training transitions and Phase 3
consumed 900,000, for 1,450,000 training simulator transitions total. The
42,289 final Validation interactions are reported separately and are not part
of the training budget.

## Frozen historical comparisons

The primary comparison is the historically engineered scalar SB3-SAC reward.
It matches the environment, 1,800-step episode cap, task and zero speed
tolerance, observation/action semantics, seeds, 300,000-transition budget,
Validation tracks, deterministic final evaluation, SAC settings and absence of
normalization. The scalar study also recorded separate milestone evaluations;
the table below uses only its 300k policies, and evaluation never entered its
replay buffer.

| Formulation | Role | RSR | Completed | Completed by 140 s | Speed-compliant, all episodes |
|---|---|---:|---:|---:|---:|
| Frozen LLM reward | primary study | 1/27 (3.7%) | 8/27 | 8/27 | 20/27 |
| Historically engineered scalar reward, SB3 SAC | matched primary reference | 5/27 (18.5%) | 5/27 | 5/27 | 27/27 |
| Binary Reward V1, SB3 SAC | contextual, otherwise matched | 0/27 | 0/27 | 0/27 | 27/27 |
| Constrained V2 | contextual only | 21/27 (77.8%) | 27/27 | 27/27 | 21/27 |

The LLM reward completed more episodes than the scalar reference, but seven of
its eight completions violated the exact speed requirement. Its final RSR was
therefore 4/27 lower (14.8 percentage points). Binary V1 confirms that the same
SAC setup with terminal binary feedback acquired no completion. Constrained V2
uses FSRL SACLag, different networks, replay, update rate, objective and
explicit costs, so its much higher RSR is context rather than a reward-only
comparison.

There are zero jointly feasible seed-track pairs between the LLM reward and the
primary scalar baseline, despite five feasible scalar episodes. A paired energy
claim is therefore impossible. The only jointly feasible Constrained-V2 pair
is seed 11 / track 3008: 0.197414 kWh for the LLM reward versus 0.165061 kWh for
Constrained V2 (ratio 1.196). One pair under a different algorithm is
insufficient to establish energy superiority or inferiority.

## Interpretation

**A. Task acquisition.** The reward produced genuinely feasible driving once,
so it can induce the canonical task, but only sparsely in this experiment.

**B. Seed robustness.** Useful behavior did not replicate beyond training seed
11, the same seed used for candidate screening. Seeds 29 and 47 completed no
route. The result is not robust across the preregistered policy seeds.

**C. Efficiency.** One feasible episode is insufficient for an efficiency
claim, and there is no jointly feasible pair with the matched scalar baseline.

This was one completed LLM reward search followed by three policy-training
replicates, not three independent LLM searches. Generation 2 did not replace
the G1 winner under the frozen ranking, so the result does not show monotonic
improvement across reflection rounds. The best G0 candidate was not separately
trained for 300k under this protocol, preventing a controlled iterative-versus-
one-shot claim. No separate final robustness threshold was preregistered, so no
new favorable threshold is introduced here. One search cannot establish that
LLMs outperform human reward engineering.

## Integrity and verification

The final manifest and all three result files agree on the winner, protocol,
task, split and budget. All eleven candidate hashes, the final reward freeze,
the protocol/ranking, the external task/metrics implementation, public
environments, historical rewards and historical benchmark artifacts remained
unchanged. Validation was not inspected until all three sequential runs had
finished. Tracks 4000--4017 remained sealed and untouched. The search remains
`CLOSED`.

Verification results:

- focused LLM/task/metrics/environment tests: 170 passed, with the same two
  obsolete Phase-1 pristine-state assertions failing;
- complete test suite: 342 passed and the same two assertions failed
  (`test_repository_history_is_pristine_and_not_started` and
  `test_phase_one_contains_no_generated_or_final_reward_files`); these tests
  still require an unstarted Phase-1 repository and are incompatible with the
  now-completed, intentionally populated search;
- Ruff: the only finding is the previously recorded immutable `g0-c03` F841
  (`budget_scale`); excluding that exact immutable candidate, all checks pass;
- `git diff --check`: passed;
- every entry in `frozen_artifacts.sha256`: passed;
- offline sdist and wheel build: passed;
- installation smoke from the built wheel: passed for both public v1
  environments, task metrics and packaged assets.

Neither known exception was edited or suppressed. The complete per-episode
data and derived statistics are in `final-results.json`;
`final-evaluation-manifest.json` preserves the runner's compact manifest.
