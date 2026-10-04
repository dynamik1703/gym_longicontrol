# Historical SAC versus SB3 SAC

This is a structural diagnostic, not a bug report. The implementations differ
in several material ways, so the historical trainer should not be treated as an
independent replication of modern SAC.

| Aspect | Historical/custom SAC | SB3 SAC 2.9 |
| --- | --- | --- |
| critic formulation | twin Q networks plus a separately learned value network | twin Q networks, no separate value network |
| target network | target value network | target twin-Q critics |
| Bellman target | `r + gamma * target_V(s')` | sampled next action with `min(target_Q1, target_Q2) - alpha * log_pi` |
| entropy | learned `log_alpha`, initialized at alpha 1, target `-action_dim` | automatic entropy coefficient and target entropy |
| V2 learning rate | 0.001 for all optimizers | default 0.0003 |
| V2 soft update | `tau=0.01` every online update | default `tau=0.005` every gradient update |
| policy/Q network | 64×64 ReLU; additional 64×64 value network | default 256×256 ReLU policy and critics |
| replay capacity | 400,000 | default 1,000,000 |
| warm-up accounting | 10,000 separate random transitions, then 300,000 online steps | 100 learning-start interactions inside the 300,000-step budget |
| update cadence | one update per online step after prefilling | one gradient step per environment step after learning starts |
| milestone handling | training is called in 50k blocks and resets the training environment per block | one uninterrupted call with external validation |
| truncation target | bootstraps across truncation by storing only `terminated` | handles time-limit truncation separately from terminal states |

Both use tanh-squashed stochastic Gaussian actions, twin critics, learned
entropy regularization, deterministic mean actions for evaluation, and one
environment at a time. None of these differences alone proves a defect. The
controlled validation curves provide the empirical evidence for deciding
whether the historical implementation was the dominant confounder.
