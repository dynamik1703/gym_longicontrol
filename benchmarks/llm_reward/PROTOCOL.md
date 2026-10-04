# LLM Reward Engineering V1 protocol

Status: **Phase 1 infrastructure frozen before any generated reward exists**.

## Research question

If dense reward engineering is useful for LongiControl, must a human perform it
manually, or can a coding LLM discover an effective scalar reward under a fixed
automated search and reflection budget?

The controlled eventual comparison is human-designed dense scalar reward versus
LLM-designed dense scalar reward. Both use the frozen Stable-Baselines3 SAC
family and the same physical task. This study does not compare SAC with another
optimizer.

## Three phases

1. **Phase 1 -- infrastructure freeze (this change):** define the information
   barrier, API, budgets, checks, ranking, reflection, history, winner freeze and
   final-evaluation gate. It contains no real LLM reward candidate and executes
   no candidate training or evaluation.
2. **Phase 2 -- fresh-session reward search:** generate, ingest and screen the
   fixed 5+3+3 candidates using Development information only.
3. **Phase 3 -- winner freeze and final evaluation:** mechanically select and
   hash one winner before opening Validation, then train it from scratch on
   three seeds and evaluate it once.

No search rule may be changed after a generated reward or screening result has
been seen. A changed protocol requires a new protocol ID.

## Canonical task and algorithm

The physical requirement is route completion within 140 seconds with maximum
measured speed excess exactly zero; energy is minimized among feasible behavior.
The authoritative evaluator remains `TaskSpecification`, `EpisodeMetrics` and
`is_feasible`. Reward return and component values are never evaluation metrics.

The learner is Stable-Baselines3 SAC with the frozen human Scalar-SAC settings:
256x256 network, learning rate `3e-4`, replay capacity 1,000,000, learning starts
at 100 interactions, batch 256, `tau=0.005`, `gamma=0.99`, train frequency 1,
one gradient step per interaction and automatic entropy tuning. There is no SAC
hyperparameter search, reward normalization or observation normalization.

## Search budget

| Generation | New candidates | Information available |
|---:|---:|---|
| 0 | 5 | sanitized task, signal whitelist and reward API only |
| 1 | 3 | same context plus two selected parents and reflections |
| 2 | 3 | same context plus two selected parents and reflections |

Maximum: **11 generated reward functions**. Generation 2 is final. There is no
candidate replacement after technical failure and no fourth generation.

Generation 0 must contain at least three rewards that differ in functional
form, not merely numerical coefficients. This is an instruction to the later
Reward Designer, not a reward template.

## Screening budget and tracks

Every valid candidate receives exactly 50,000 native simulator transitions with
SAC training seed 11. Deterministic policy evaluation occurs at 10k, 25k and
50k on Development tracks 2000--2008 only. Evaluation transitions do not enter
replay and do not extend the training budget.

During search, the following are forbidden for generation, reflection, ranking,
selection, hyperparameters and stopping decisions:

- Validation tracks 3000--3008;
- historical exploratory tracks 1000--1008;
- sealed paper-final tracks 4000--4017;
- results, trajectories and reward implementations from completed studies.

The final winner is frozen before Validation is opened. Paper-final tracks stay
untouched throughout this protocol.

## Information barrier

The fresh Reward Designer receives only `context/TASK.md`,
`context/ALLOWED_SIGNALS.md`, `context/REWARD_API.md` and generation-specific
instructions. Later generations additionally receive only the two mechanically
selected parent sources, their visible rationales and sanitized aggregate
Development reflections.

It never receives historical benchmark results, known reward formulas,
Constrained deadline-credit/slack code, prior failure diagnoses, per-track
identities, Validation information, sealed-track metadata, optimal trajectories,
optimal actions or hidden evaluator state. Generation packages are scanned for
these leak classes before being written.

## Candidate API and checks

The immutable `RewardContext` exposes only the 21 physical fields documented in
`ALLOWED_SIGNALS.md`. Candidates implement a deterministic, per-transition
`compute_reward(ctx) -> RewardOutput`, where `reward` and every diagnostic
component are finite scalars.

Before training, the harness parses the source AST, rejects non-whitelisted
imports and attributes, file/network/process/environment access, randomness,
dynamic evaluation, global/nonlocal state, nonlocal assignment targets, track
identifiers and historical research references. It then executes deterministic
smoke contexts in a restricted namespace and verifies exact repeated outputs.
This is an experiment-integrity barrier, not a claim of operating-system-grade
sandbox security.

Ingestion assigns IDs `g0-c01` through `g2-c03`, hashes the exact UTF-8 source,
stores the visible rationale and metadata, and records the attempt in
`search_history.json`. Screening accepts only a hash-matching, successfully
ingested candidate.

## Retry and human-intervention policy

One syntax/API repair request is allowed after a failed first check. Its feedback
contains only checker errors, and the repair must preserve the intended reward
design. If the repaired source remains invalid, its original slot becomes
`technical_failure`; it is not replaced.

Humans may repair harness bugs, restart technically invalid infrastructure runs
and document failures. Humans may not edit reward formulas, propose coefficients
after seeing behavior, choose parents or winners, add behavioral hints, extend a
candidate budget or add generations. Human reward edits and human-selected
coefficients during search are recorded as zero.

## Reflection schema

Each reflection contains only aggregate Development information:

- candidate ID, generation and source hash;
- reward-component count/mean/std/min/max;
- episode-return and episode-length statistics;
- successful training-episode count and first-success step;
- Development RSR trajectory at 10k/25k/50k;
- completion and completed-episode deadline compliance;
- speed compliance and violation severity on completed or substantially
  progressing trajectories;
- exclusive failure-mode counts;
- travel-time and speed-violation distributions;
- feasible-episode energy;
- normalized progress for incomplete episodes.

Per-track identities, source paths and forbidden split information are removed.
Reflection values never alter the training reward automatically.

## Exact lexicographic ranking

Candidates are sorted deterministically by:

1. Development RSR, descending;
2. completion rate, descending;
3. median normalized route progress for incomplete episodes, descending;
4. deadline compliance among completed episodes, descending;
5. speed compliance among completed or at least 50%-progress trajectories,
   descending;
6. mean integrated speed violation on that relevant subset, ascending;
7. mean energy among feasible episodes only, ascending;
8. candidate ID, ascending.

Missing speed evidence is worst (`+infinity` severity), and missing feasible
energy is worst (`+infinity`). Earlier fields always dominate later fields.
Thus standstill cannot win due to speed compliance or low energy, and energy
cannot compensate for lower feasibility. This is a search-selection rule, not
an agent reward.

After each completed generation, the top two candidates across all candidates
seen so far become the next parents. There is no human override. If fewer than
two candidates remain valid, the search cannot advance and ends as a technical
failure under this protocol. The best Generation-0 candidate is recorded
separately as the zero-shot control.

## Search history and effort accounting

`search_history.json` records protocol identity/hash, status, every slot and
attempt, generation, parents, exact source hash, screening provenance, reflection
paths, ranks, parent selections, technical failures, repair count and winner.
It also records generated-candidate count, started generations, screening RL
transitions, final-training transitions, human edits, human-selected coefficients
and the number of explicit reflection metrics.

Candidate metadata additionally supports source-line, numeric-constant,
component-count and AST-complexity accounting. Reward-generation compute and
policy-training compute remain separate; no subjective combined engineering
score is defined.

## Winner freeze and final evaluation

After every Generation-2 slot is resolved, the same ranking selects one winner.
The freeze command copies its exact source to `FINAL_REWARD.py`, writes protocol
and source hashes, and changes search status to `CLOSED`. Ingestion, screening,
parent selection and further generations reject a closed history.

Only the hash-matching winner may then be trained from scratch with SB3 SAC for
300,000 transitions on each seed 11, 29 and 47. Only after this freeze may the
runner evaluate Validation tracks 3000--3008 using physical RSR, completion,
deadline and speed compliance, failure reasons, travel time, feasible energy and
speed-violation metrics. Once opened, Validation cannot trigger reward changes.

The paper-final tracks remain sealed. Beating one historical baseline would
support only the statement that, under this fixed automated search budget, the
generated reward produced a stronger policy than that frozen baseline; it would
not establish that LLMs are generally better reward engineers than humans.

