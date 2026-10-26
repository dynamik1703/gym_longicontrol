# Future masked RL: design only, no training authorization
V3 is not ready for a masked-RL experiment: the study storage projection fails and
Development authority/progress remain untested. Total future RL compute feasibility
is separately UNESTABLISHED. No learner was trained or installed in this package.

Prepared interface: ordinary causal observation, declared physical memory,
position/time/remaining-distance/deadline information and current 21-action mask;
output one certified native index. Apply mask before sampling. Invalid logits may
not contribute to probabilities, log-probabilities, entropy or relevant targets.
safe_action_mask.masked_log_probabilities returns zero invalid probability and
minus-infinity invalid log probability; entropy sums only positive valid mass.
A future algorithm must select valid target values explicitly, avoiding 0*(-inf)
or 0*NaN arithmetic. Final full native certification remains mandatory.
No hidden route, reset seed or Oracle is a policy input. Speed safety and deadline
optimization remain separate. No RL gradient or training-transition code ran.

A scenario of 300000 native transitions per seed11/29/47 means 900000 for one arm;
the controlled no-pre-shield / inherited-post-shield / Banker-pre-mask comparison
means 9 runs and 2700000 total transitions. This is not a frozen learner-budget
approval. rl_projection.json preserves every numerical projection and caveat.
Mean measured mask+final component projects
1.345h per three-seed masked arm;
observed-worst component projects
5.097h.
Both exclude learner updates and cannot establish total training within 24h.

Non-learning controller/native-evidence pipeline proxies, including independent
full-grid replay and I/O, give these separate 900k-step engineering scenarios:

| Physical pipeline proxy | Uniform measured speed-band mean | Worst measured speed-band |
|---|---:|---:|
| CAUSAL60 | 1.154h | 1.655h |
| LEGACY_NATIVE_POST_GREEDY | 1.351h | 1.700h |
| BANKERS_GREEDY | 17.037h | 31.556h |
| BANKERS_MASKED_CAUSAL60 | 16.867h | 31.434h |

The future three-arm proxy totals
19.372h optimistically and
34.789h conservatively,
BEFORE unknown learner inference/updates, actual replay/target handling,
checkpoint/recovery, diagnostics and evaluation. These are not same-RL measured
throughputs: CAUSAL60 and+1 post-shield workloads are merely declared physical
pipeline proxies, not an estimate that all learner proposals cost the same.
No finite update-throughput measurement exists, so a defensible numeric end-to-end
RL runtime is unavailable. The formulas remain parameterized by those unknowns.

Compute one mask per actual causal successor and store its exact source/config,
observation/memory identity with replay. Reuse it only if those identities are
unchanged; do not recompute a second full grid inside every Bellman update.
Mask application and valid-target handling still require algorithm-specific tests.
No successor-mask cache or checkpoint recovery is claimed tested as a learner.
STOP: no RL, Validation, Paper or subsequent experiment is started.
