# Seed 11 mechanism audit

Validation E−T differences: RSR `+3`, completion
`+0`, deadline `+0`, speed
`+3`.

## Transition-aligned timeline

Rows marked Δ are EPISODIC minus TRANSITIONAL. Diagnostic counts are
RSR/completion/deadline/speed, each out of 18.

| Step | C | Updates | Episodes | Tasks | Unique routes/tasks | PID | lambda S/D | alpha | diagnostic RSR/C/D/S | regime |
|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 0 | E | 0 | 0 | 0 | 0/0 | 0 | 0.000/0.000 | 1.000000 | — | — |
| 0 | T | 0 | 0 | 0 | 0/0 | 0 | 0.000/0.000 | 1.000000 | — | — |
| 0 | Δ | +0 | +0 | +0 | +0/+0 | +0 | +0.000/+0.000 | +0.000000 | — | E-T |
| 50,000 | E | 4,962 | 35 | 36 | 36/36 | 35 | 1.222/4.967 | 0.234607 | 0/18/18/0 | COMPLETES_WITH_SPEED_VIOLATION |
| 50,000 | T | 5,000 | 32 | 33 | 33/33 | 32 | 1.120/1.798 | 0.236332 | 7/18/14/7 | COMPLETES_WITH_SPEED_VIOLATION |
| 50,000 | Δ | -38 | +3 | +3 | +3/+3 | +3 | +0.102/+3.169 | -0.001725 | -7/+0/+4/-7 | E-T |
| 100,000 | E | 9,866 | 73 | 74 | 74/74 | 73 | 43.919/0.772 | 0.064882 | 5/6/5/18 | CRAWLING_OR_PARTIAL |
| 100,000 | T | 10,000 | 71 | 72 | 72/72 | 71 | 2.412/0.535 | 0.055276 | 8/10/8/15 | MIXED |
| 100,000 | Δ | -134 | +2 | +2 | +2/+2 | +2 | +41.507/+0.238 | +0.009606 | -3/-4/-3/+3 | E-T |
| 150,000 | E | 14,999 | 119 | 120 | 120/120 | 119 | 2.035/0.851 | 0.015058 | 8/17/10/11 | MIXED |
| 150,000 | T | 15,000 | 114 | 115 | 115/115 | 114 | 1.619/0.623 | 0.012644 | 8/14/8/13 | MIXED |
| 150,000 | Δ | -1 | +5 | +5 | +5/+5 | +5 | +0.416/+0.228 | +0.002415 | +0/+3/+2/-2 | E-T |
| 200,000 | E | 19,922 | 155 | 156 | 155/155 | 155 | 2.082/4.111 | 0.004276 | 6/18/18/6 | COMPLETES_WITH_SPEED_VIOLATION |
| 200,000 | T | 20,000 | 158 | 159 | 158/158 | 158 | 1.682/1.486 | 0.003293 | 13/18/13/18 | FEASIBLE |
| 200,000 | Δ | -78 | -3 | -3 | -3/-3 | -3 | +0.400/+2.624 | +0.000982 | -7/+0/+5/-12 | E-T |
| 250,000 | E | 24,925 | 195 | 196 | 194/194 | 195 | 2.203/1.007 | 0.001016 | 2/16/2/18 | COMPLETES_TOO_SLOW |
| 250,000 | T | 25,000 | 193 | 194 | 192/192 | 193 | 1.938/1.481 | 0.001022 | 13/18/15/13 | FEASIBLE |
| 250,000 | Δ | -75 | +2 | +2 | +2/+2 | +2 | +0.266/-0.474 | -0.000006 | -11/-2/-13/+5 | E-T |
| 300,000 | E | 30,000 | 243 | 243 | 240/240 | 243 | 2.186/1.075 | 0.000562 | 12/18/12/17 | FEASIBLE |
| 300,000 | T | 30,000 | 233 | 233 | 231/231 | 233 | 1.737/1.002 | 0.000677 | 9/16/12/15 | MIXED |
| 300,000 | Δ | +0 | +10 | +10 | +9/+9 | +10 | +0.449/+0.073 | -0.000116 | +3/+2/+0/+2 | E-T |

## Mechanism scorecard

| H | Supporting evidence | Evidence against / limit | Completeness | Direction | Confidence |
|:--:|:--|:--|:--|:--|:--|
| H1 | EPISODIC has greater speed/deadline jump volatility and applies all updates immediately after PID; this can explain stronger feedback, but the same contrast exists in losing seeds. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | HIGH | EPISODIC immediate block exposure and generally higher volatility | HIGH for timing; LOW for causal attribution |
| H2 | EPISODIC makes zero actor updates during collection versus about 129 per TRANSITIONAL episode; fixed-policy trajectories may reduce within-task drift, but this does not identify the favorable sign. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | HIGH | EPISODIC fixed; TRANSITIONAL changes within episode | HIGH |
| H3 | All EPISODIC updates see a complete just-finished episode versus a tiny terminal fraction under TRANSITIONAL; sampled indices are absent. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | PARTIAL | EPISODIC batches; TRANSITIONAL streams updates | MEDIUM |
| H4 | EPISODIC completes 10 more episodes and has shorter episodes, a direction compatible with its +3 result here only. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | HIGH | trajectory-dependent | HIGH |
| H5 | Both oscillate; EPISODIC finishes diagnostic FEASIBLE (12/18 RSR) while TRANSITIONAL is MIXED (9/18), matching Validation's +3. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | HIGH | trajectory-dependent | HIGH |
| H6 | Final alpha is similar (EPISODIC slightly lower); exploration does not provide a clear seed-11 advantage. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | HIGH at episodes/checkpoints | trajectory-dependent | MEDIUM |
| H7 | Checkpoint actor and critic losses decline smoothly in both; no unique collapse precedes the +3 result. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | PARTIAL snapshots; no per-update history/targets | trajectory-dependent | LOW |

## Interpretation

EPISODIC may help by turning complete, stationary-policy episodes into coherent immediate PID-conditioned update blocks. It ends with better speed compliance and the same Validation completion/deadline counts. However, identical schedule amplification hurts elsewhere, so this is a seed-11-consistent interaction rather than an isolated cause.

The complete per-episode PID sequence, costs, multiplier jumps, update counts,
actor-hash evidence, task fingerprints, and every episode-index E−T delta are in
`results.json`. Episode-index summaries at 25/50/100/150/200 episodes are also
recorded there, avoiding transition-count/episode-count conflation.
