# Requirement-Conditioned RL V1

This separately preregistered benchmark asks whether one frozen-capacity FSRL
SACLag policy can observe an episode deadline and change its driving behavior.
The public environment remains eight-dimensional; a benchmark-only wrapper adds
normalized absolute deadline and elapsed time.

Training requirements use physics-selected margins of 20/40/60 seconds above
`T_min(start)`. Margins 30/50 seconds are withheld for zero-shot interpolation.
See `PROTOCOL.md` for frozen metrics and gates.

Research dependencies are the same pinned environment used by Constrained V2.
The main runner is:

```bash
PYTHONPATH=runs/constrained-rl-deps:src:. python -m \
  benchmarks.requirement_conditioned.experiment \
  --output-dir runs/requirement-conditioned-balanced-20261003 \
  --workers 3 --threads 2
```

Generate the frozen analysis, deterministic trajectories, and plots with:

```bash
PYTHONPATH=src:. python -m benchmarks.requirement_conditioned.analysis \
  runs/requirement-conditioned-balanced-20261003 \
  --output benchmarks/requirement_conditioned/results.json

PYTHONPATH=runs/constrained-rl-deps:src:. python -m \
  benchmarks.requirement_conditioned.trajectories \
  runs/requirement-conditioned-balanced-20261003 \
  --output benchmarks/requirement_conditioned/trajectories.json

PYTHONPATH=src:. python -m benchmarks.requirement_conditioned.plot_results \
  benchmarks/requirement_conditioned/results.json \
  --results-dir runs/requirement-conditioned-balanced-20261003 \
  --trajectories benchmarks/requirement_conditioned/trajectories.json \
  --output-dir benchmarks/requirement_conditioned/plots
```

The analyzer rejects incomplete seeds/checkpoints, mismatched configuration
hashes, unbalanced completed-episode exposure, and any occurrence of a sealed
4000--4017 track.

No requirement-conditioned result should be interpreted without both physical
RSR and the preregistered within-track sensitivity analysis.

## Result

The completed study is classified as **Gate D**. Final pooled RSR is 48/135
(35.6%): the 20-s tight margin reaches 5/27, while the 60-s margin reaches
14/27. Travel time responds strongly and interpolates smoothly, but feasible
energy does not decrease systematically and the separate canonical 140-s result
falls to 12/27 versus frozen Constrained V2's 21/27. See [RESULTS.md](RESULTS.md)
for the full analysis and `results.json` for machine-readable values.
