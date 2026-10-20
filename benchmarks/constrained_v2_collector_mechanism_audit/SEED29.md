# Seed 29 mechanism audit

Validation E−T differences: RSR `-4`, completion
`+7`, deadline `-3`, speed
`-5`.

## Transition-aligned timeline

Rows marked Δ are EPISODIC minus TRANSITIONAL. Diagnostic counts are
RSR/completion/deadline/speed, each out of 18.

| Step | C | Updates | Episodes | Tasks | Unique routes/tasks | PID | lambda S/D | alpha | diagnostic RSR/C/D/S | regime |
|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 0 | E | 0 | 0 | 0 | 0/0 | 0 | 0.000/0.000 | 1.000000 | — | — |
| 0 | T | 0 | 0 | 0 | 0/0 | 0 | 0.000/0.000 | 1.000000 | — | — |
| 0 | Δ | +0 | +0 | +0 | +0/+0 | +0 | +0.000/+0.000 | +0.000000 | — | E-T |
| 50,000 | E | 4,988 | 37 | 38 | 38/38 | 37 | 1.317/0.508 | 0.236364 | 0/0/0/18 | CRAWLING_OR_PARTIAL |
| 50,000 | T | 5,000 | 27 | 28 | 28/28 | 27 | 0.000/7.314 | 0.223319 | 0/0/0/18 | CRAWLING_OR_PARTIAL |
| 50,000 | Δ | -12 | +10 | +10 | +10/+10 | +10 | +1.317/-6.806 | +0.013045 | +0/+0/+0/+0 | E-T |
| 100,000 | E | 9,959 | 78 | 79 | 79/79 | 78 | 2.749/0.743 | 0.062152 | 7/9/7/17 | MIXED |
| 100,000 | T | 10,000 | 62 | 63 | 63/63 | 62 | 0.702/1.052 | 0.053571 | 6/10/7/16 | MIXED |
| 100,000 | Δ | -41 | +16 | +16 | +16/+16 | +16 | +2.047/-0.309 | +0.008581 | +1/-1/+0/+1 | E-T |
| 150,000 | E | 14,874 | 117 | 118 | 117/117 | 117 | 2.022/0.848 | 0.016056 | 5/16/11/12 | MIXED |
| 150,000 | T | 15,000 | 109 | 110 | 109/109 | 109 | 1.237/1.197 | 0.017530 | 6/18/15/7 | COMPLETES_WITH_SPEED_VIOLATION |
| 150,000 | Δ | -126 | +8 | +8 | +8/+8 | +8 | +0.785/-0.349 | -0.001475 | -1/-2/-4/+5 | E-T |
| 200,000 | E | 19,981 | 157 | 158 | 157/157 | 157 | 2.056/1.030 | 0.004657 | 7/16/7/18 | MIXED |
| 200,000 | T | 20,000 | 150 | 151 | 150/150 | 150 | 1.313/1.280 | 0.005026 | 7/9/7/18 | MIXED |
| 200,000 | Δ | -19 | +7 | +7 | +7/+7 | +7 | +0.743/-0.251 | -0.000370 | +0/+7/+0/+0 | E-T |
| 250,000 | E | 24,824 | 195 | 196 | 194/194 | 195 | 2.099/1.104 | 0.001406 | 7/18/9/11 | MIXED |
| 250,000 | T | 25,000 | 193 | 194 | 192/192 | 193 | 1.542/1.405 | 0.001699 | 7/16/14/9 | MIXED |
| 250,000 | Δ | -176 | +2 | +2 | +2/+2 | +2 | +0.557/-0.301 | -0.000293 | +0/+2/-5/+2 | E-T |
| 300,000 | E | 30,000 | 236 | 236 | 231/231 | 236 | 2.094/1.174 | 0.000583 | 7/18/8/13 | MIXED |
| 300,000 | T | 30,000 | 238 | 238 | 233/233 | 238 | 1.401/1.575 | 0.001187 | 6/11/7/16 | MIXED |
| 300,000 | Δ | +0 | -2 | -2 | -2/-2 | -2 | +0.693/-0.402 | -0.000604 | +1/+7/+1/-3 | E-T |

## Mechanism scorecard

| H | Supporting evidence | Evidence against / limit | Completeness | Direction | Confidence |
|:--:|:--|:--|:--|:--|:--|
| H1 | EPISODIC again has higher multiplier volatility and immediate PID exposure, but that treatment-level fact does not uniquely explain -4. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | HIGH | EPISODIC immediate block exposure and generally higher volatility | HIGH for timing; LOW for causal attribution |
| H2 | The large within-episode update contrast is direct, yet identical in kind to seed 11 where the outcome reverses. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | HIGH | EPISODIC fixed; TRANSITIONAL changes within episode | HIGH |
| H3 | Schedule-level freshness differs strongly; actual samples and ages were not logged, preventing attribution. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | PARTIAL | EPISODIC batches; TRANSITIONAL streams updates | MEDIUM |
| H4 | Episode counts and mean lengths are almost equal (236 versus 238), arguing against task exposure as the main seed-29 cause. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | HIGH | trajectory-dependent | HIGH |
| H5 | Both end MIXED on diagnostics; Validation shows EPISODIC completes more but loses deadline and speed compliance, an aggressive tradeoff. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | HIGH | trajectory-dependent | HIGH |
| H6 | EPISODIC ends at about half the TRANSITIONAL alpha, consistent with less late exploration but not causally identified. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | HIGH at episodes/checkpoints | trajectory-dependent | MEDIUM |
| H7 | TRANSITIONAL reward/constraint critic losses are much lower at the retained checkpoints, but snapshot losses are scale-sensitive and not direct error estimates. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | PARTIAL snapshots; no per-update history/targets | trajectory-dependent | LOW |

## Interpretation

TRANSITIONAL may help by avoiding large episode-end amplification and retaining more late entropy. EPISODIC completes seven more Validation tasks but loses three deadline and five speed-compliance cases, suggesting an aggressive constraint tradeoff rather than a general completion failure. The diagnostic endpoint does not reproduce the full Validation gap, so attribution remains low-confidence.

The complete per-episode PID sequence, costs, multiplier jumps, update counts,
actor-hash evidence, task fingerprints, and every episode-index E−T delta are in
`results.json`. Episode-index summaries at 25/50/100/150/200 episodes are also
recorded there, avoiding transition-count/episode-count conflation.
