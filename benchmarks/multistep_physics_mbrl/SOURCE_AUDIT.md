# Primary-source and implementation audit

Audit date: 2026-10-13. External repositories were inspected outside this repository;
none are vendored here. `upstream.json` records machine-readable references.

## Model-Based Value Expansion

- Feinberg et al., *Model-Based Value Estimation for Efficient Model-Free
  Reinforcement Learning*, arXiv `1803.00101`.
- Primary paper: <https://arxiv.org/abs/1803.00101>

MVE expands a value target through model-predicted transitions and then bootstraps a
learned value. This study adopts that defining operation, but uses the exact
LongiControl dynamics and a soft constrained off-policy critic. No paper code is
imported, and the frozen two-real-step prefix is a project-specific requirement.

## Revisiting Model-Based Value Expansion

- Palenicek, Lutter, and Peters, *Revisiting Model-based Value Expansion*, arXiv
  `2203.14660`.
- Primary paper: <https://arxiv.org/abs/2203.14660>

This work is relevant because it separates value-expansion effects from learned-model
error and examines the use of true dynamics. Long-horizon compounding model error is
not the variable here: the benchmark's exact deterministic physics is used and its
scalar/vectorized parity is tested. Value/optimization effects can still make MVE
harmful, so the protocol does not assume improvement.

## MBPO

- Janner et al., *When to Trust Your Model: Model-Based Policy Optimization*, arXiv
  `1906.08253`: <https://arxiv.org/abs/1906.08253>
- Official code: <https://github.com/JannerM/mbpo>, audited revision
  `ac694ff9f1ebb789cc5b3f164d9d67f93ed8f129`.

MBPO branches short model rollouts from real data and trains on replay containing
model-generated transitions. The completed LongiControl H=1 arm used that
Dyna/MBPO-style pattern (128 real plus 128 synthetic samples, with two-step and
one-step targets mixed). This study deliberately does not: model states never enter a
replay sampling distribution, and one real batch receives one coherent expanded
target.

## TD-MPC2 and online planning

- Hansen et al., *TD-MPC2: Scalable, Robust World Models for Continuous Control*,
  arXiv `2310.16828`: <https://arxiv.org/abs/2310.16828>
- ICLR 2024 paper: <https://proceedings.iclr.cc/paper_files/paper/2024/hash/cf73d57e17eacf5b2cc004a2a6271e0a-Abstract-Conference.html>
- Official code: <https://github.com/nicklashansen/tdmpc2>, audited revision
  `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`.

TD-MPC2 learns latent dynamics and uses trajectory optimization for online control.
It is contextual only. This benchmark does not implement TD-MPC2, CEM, MPC, action
sequence optimization, or any online model query.

## Pinned constrained learner

- FSRL official code: <https://github.com/liuzuxin/FSRL>, exact revision
  `e056fc9498d5d037869533da7cf976acf462f918`.
- Tianshou `0.5.1`.

The audit inspected `fsrl/policy/sac_lag.py`, `fsrl/policy/base_policy.py`, and
`fsrl/trainer/offpolicy.py` at that revision. The relevant behavior is:

1. `process_fn` asks the replay buffer for an `n_step=2` return.
2. `get_metrics` supplies objective, speed cost, then deadline cost.
3. Each target critic predicts the minimum of its twin target heads.
4. `_target_q` subtracts `alpha * log_prob` for every critic, including both cost
   critics.
5. a true `terminated` transition masks the target value; a time-limit `truncated`
   transition ends reward traversal but retains a bootstrap.
6. `learn` updates critics, then the actor and automatic entropy coefficient from
   `batch.obs`, then softly updates target critics.
7. PID multipliers are updated only by `pre_update_fn(stats_train["cost"])`.

The M=0 path calls the untouched `policy.update(batch_size, real_buffer)` and is tested
bit-for-bit against it. The M>0 path changes only critic target construction; it calls
the same `learn`, so actor inputs and all optimizer semantics remain pinned.
