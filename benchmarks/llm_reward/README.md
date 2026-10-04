# LLM Reward Engineering V1

This directory contains the frozen Phase-1 harness for a small Eureka-style
reward search. It contains **no generated reward candidate and no LLM reward
result**. Read `PROTOCOL.md` before opening the search.

## Phase 2 workflow (future fresh session)

Start a copy of the pristine history only when the real search begins:

```bash
python -m benchmarks.llm_reward.history start
```

Create the Generation-0 package and provide that single Markdown file to a
fresh Codex session:

```bash
python -m benchmarks.llm_reward.generation_package \
  --generation 0 \
  --output runs/llm-reward-search/generation-0-package.md
```

For each returned source, store its visible rationale in a text file and ingest
it. Candidate IDs are assigned by the harness:

```bash
python -m benchmarks.llm_reward.candidate_validation incoming/reward.py \
  --generation 0 \
  --rationale-file incoming/rationale.txt
```

If this reports a checker failure, one repair may be ingested with
`--repair-candidate-id`. A second failure closes that slot permanently.

Screen and reflect each valid candidate:

```bash
python -m benchmarks.llm_reward.screening g0-c01 \
  --output-root runs/llm-reward-search/screening
python -m benchmarks.llm_reward.feedback g0-c01 \
  --output-root runs/llm-reward-search/reflections
```

After all five Generation-0 slots are resolved, select parents mechanically and
export the next fresh-session package:

```bash
python -m benchmarks.llm_reward.ranking select-parents --for-generation 1
python -m benchmarks.llm_reward.generation_package \
  --generation 1 \
  --output runs/llm-reward-search/generation-1-package.md
```

Ingest exactly three new candidates using the two IDs printed by
`select-parents` as repeated `--parent` arguments, then screen and reflect them.
Repeat once for Generation 2:

```bash
python -m benchmarks.llm_reward.ranking select-parents --for-generation 2
python -m benchmarks.llm_reward.generation_package \
  --generation 2 \
  --output runs/llm-reward-search/generation-2-package.md
```

There is no Generation 3. After all 11 slots are resolved, freeze exactly one
winner:

```bash
python -m benchmarks.llm_reward.ranking freeze-winner
```

Only after that command closes the search may Phase 3 train the immutable winner
and open the final evaluation split:

```bash
python -m benchmarks.llm_reward.final_evaluation \
  --output-root runs/llm-reward-final
```

The final command is intentionally impossible while history is `NOT_STARTED` or
`OPEN`, when the source hash differs, or when any non-winning source is supplied.
Do not run any Phase-2 or Phase-3 command during the infrastructure freeze.
