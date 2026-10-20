# Seed 47 mechanism audit

Validation E−T differences: RSR `-7`, completion
`-7`, deadline `-6`, speed
`-3`.

## Transition-aligned timeline

Rows marked Δ are EPISODIC minus TRANSITIONAL. Diagnostic counts are
RSR/completion/deadline/speed, each out of 18.

| Step | C | Updates | Episodes | Tasks | Unique routes/tasks | PID | lambda S/D | alpha | diagnostic RSR/C/D/S | regime |
|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 0 | E | 0 | 0 | 0 | 0/0 | 0 | 0.000/0.000 | 1.000000 | — | — |
| 0 | T | 0 | 0 | 0 | 0/0 | 0 | 0.000/0.000 | 1.000000 | — | — |
| 0 | Δ | +0 | +0 | +0 | +0/+0 | +0 | +0.000/+0.000 | +0.000000 | — | E-T |
| 50,000 | E | 4,993 | 28 | 29 | 29/29 | 28 | 3.988/0.694 | 0.223483 | 0/0/0/18 | CRAWLING_OR_PARTIAL |
| 50,000 | T | 5,000 | 27 | 28 | 28/28 | 27 | 0.000/8.873 | 0.223442 | 0/0/0/18 | CRAWLING_OR_PARTIAL |
| 50,000 | Δ | -7 | +1 | +1 | +1/+1 | +1 | +3.988/-8.179 | +0.000041 | +0/+0/+0/+0 | E-T |
| 100,000 | E | 9,898 | 66 | 67 | 67/67 | 66 | 21.855/1.059 | 0.059746 | 5/7/5/18 | CRAWLING_OR_PARTIAL |
| 100,000 | T | 10,000 | 55 | 56 | 56/56 | 55 | 0.000/4.607 | 0.050644 | 0/0/0/18 | CRAWLING_OR_PARTIAL |
| 100,000 | Δ | -102 | +11 | +11 | +11/+11 | +11 | +21.855/-3.547 | +0.009103 | +5/+7/+5/+0 | E-T |
| 150,000 | E | 14,972 | 104 | 105 | 104/104 | 104 | 1.771/7.313 | 0.019158 | 5/18/18/5 | COMPLETES_WITH_SPEED_VIOLATION |
| 150,000 | T | 15,000 | 83 | 84 | 84/84 | 83 | 1.583/2.302 | 0.012838 | 0/7/5/4 | CRAWLING_OR_PARTIAL |
| 150,000 | Δ | -28 | +21 | +21 | +20/+20 | +21 | +0.189/+5.012 | +0.006320 | +5/+11/+13/+1 | E-T |
| 200,000 | E | 19,976 | 146 | 147 | 146/146 | 146 | 1.613/2.944 | 0.004756 | 4/18/18/4 | COMPLETES_WITH_SPEED_VIOLATION |
| 200,000 | T | 20,000 | 125 | 126 | 125/125 | 125 | 0.904/2.149 | 0.016029 | 11/18/15/12 | FEASIBLE |
| 200,000 | Δ | -24 | +21 | +21 | +21/+21 | +21 | +0.709/+0.795 | -0.011273 | -7/+0/+3/-8 | E-T |
| 250,000 | E | 24,947 | 190 | 191 | 189/189 | 190 | 1.710/1.426 | 0.001619 | 10/16/10/17 | FEASIBLE |
| 250,000 | T | 25,000 | 160 | 161 | 160/160 | 160 | 0.832/3.778 | 0.004033 | 8/18/18/8 | COMPLETES_WITH_SPEED_VIOLATION |
| 250,000 | Δ | -53 | +30 | +30 | +29/+29 | +30 | +0.879/-2.351 | -0.002414 | +2/-2/-8/+9 | E-T |
| 300,000 | E | 30,000 | 236 | 236 | 234/234 | 236 | 1.721/1.459 | 0.000503 | 2/8/6/9 | CRAWLING_OR_PARTIAL |
| 300,000 | T | 30,000 | 206 | 206 | 204/204 | 206 | 1.390/2.395 | 0.004396 | 9/15/10/14 | MIXED |
| 300,000 | Δ | +0 | +30 | +30 | +30/+30 | +30 | +0.331/-0.936 | -0.003894 | -7/-7/-4/-5 | E-T |

## Mechanism scorecard

| H | Supporting evidence | Evidence against / limit | Completeness | Direction | Confidence |
|:--:|:--|:--|:--|:--|:--|
| H1 | EPISODIC's immediate PID exposure and higher speed-lambda volatility coexist with a late collapse; deadline volatility itself is similar. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | HIGH | EPISODIC immediate block exposure and generally higher volatility | HIGH for timing; LOW for causal attribution |
| H2 | TRANSITIONAL performs about 145 actor updates during an average episode and recovers late; whether nonstationarity causes recovery is not separable from replay and PID timing. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | HIGH | EPISODIC fixed; TRANSITIONAL changes within episode | HIGH |
| H3 | The schedules sharply differ, but missing sample indices make actual freshness and episode amplification unmeasurable. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | PARTIAL | EPISODIC batches; TRANSITIONAL streams updates | MEDIUM |
| H4 | EPISODIC exposes 30 more completed tasks and shorter episodes yet loses by 7, directly contradicting a simple diversity/exposure benefit. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | HIGH | trajectory-dependent | HIGH |
| H5 | EPISODIC leads at 100k-150k, is FEASIBLE-dominant at 250k, then falls to CRAWLING/PARTIAL at 300k; TRANSITIONAL recovers after 150k. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | HIGH | trajectory-dependent | HIGH |
| H6 | TRANSITIONAL alpha rebounds after 150k and ends about 8.7x EPISODIC; this is consistent with recovery, not proof that entropy caused it. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | HIGH at episodes/checkpoints | trajectory-dependent | MEDIUM |
| H7 | EPISODIC critic losses exceed TRANSITIONAL snapshots while behavior later degrades, but no per-update trace or target statistics remain. | The same collector-level contrast occurs in seeds with the opposite RSR direction, or the required fine-grained evidence was not retained. | PARTIAL snapshots; no per-update history/targets | trajectory-dependent | LOW |

## Interpretation

TRANSITIONAL may help through late recovery: its alpha rebounds, behavior moves out of prolonged crawling after 150k, and the final diagnostic mirrors the Validation advantage. EPISODIC briefly leads and is feasible-dominant at 250k before collapsing by 300k. That non-monotone path is evidence for coupled trajectory dependence, not a single collector mechanism.

The complete per-episode PID sequence, costs, multiplier jumps, update counts,
actor-hash evidence, task fingerprints, and every episode-index E−T delta are in
`results.json`. Episode-index summaries at 25/50/100/150/200 episodes are also
recorded there, avoiding transition-count/episode-count conflation.
