# Exact native categorical targets

Let `S_m(x)` be all 21 actions for RAW and POST, and the certified set for PRE.
For valid actions, `pi(a|x) = exp(z_a) / sum_{b in S_m(x)} exp(z_b)`; invalid
actions have probability exactly zero. Invalid actions contribute nothing to
any entropy or Q expectation. The implementation returns zero in invalid log
probability slots solely to evaluate `0 log 0` safely; mathematical log pi in
those slots is minus infinity.

One transition is 0.1 s. `gamma_native = 0.99 ** 0.1`. Continuation `D` is this
discount except at true termination or environment TimeLimit, where it is zero.
A final 50k collection-budget cutoff is separately labeled; it is not a task
failure and keeps the ordinary continuing target if the environment continues.

`y_R = r + D sum_{a in S_m(x')} pi(a|x') [min_j Qbar_j(x',a) - alpha log pi(a|x')]`

`y_c,k = c_k + D sum_{a in S_m(x')} pi(a|x') Qbar_c,k(x',a)`

The current and successor supports have separate exact causal-state digests.
PRE targets use the stored successor mask. RAW and POST target distributions
sum over all proposals, including proposals corrected by POST during execution.
Q_POST(x,a) therefore models the transition induced by correction of proposal a;
the executed action never replaces the proposed replay action.

Entropy `H = -sum pi log pi`; target `H*(x) = 0.98 log |S_m(x)|`. The fixed
fraction keeps broad exploration while respecting the available support; it
was chosen before any policy-performance results. Singleton support gives H=H*=0.
Temperature loss is `log_alpha * stop_gradient(H-H*)`, averaged across states.
No entropy target was tuned. Synthetic arithmetic, terminal target, singleton,
all-valid and invalid-support tests passed in the failed preflight run.
