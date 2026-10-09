# Primary-source audit

Audit date: 2026-10-09. The machine-readable URLs, licenses and revisions are in
`upstream.json`. External repositories were inspected in temporary directories and are
not vendored or committed.

## MBPO

- Paper: *When to Trust Your Model: Model-Based Policy Optimization*, arXiv
  `1906.08253`.
- Official code: `https://github.com/jannerm/mbpo` at
  `ac694ff9f1ebb789cc5b3f164d9d67f93ed8f129`.
- Inspected implementation: `mbpo/algorithms/mbpo.py`, probabilistic ensemble model,
  fake environment, and task configuration files.

Source-faithful concepts adopted here are a Gaussian dynamics ensemble, seven members,
five elites selected by holdout loss, member-specific bootstrapping, input
standardization, bounded learned log variance, Adam at `1e-3`, 20% holdout, early
stopping, periodic model refresh, short branched rollouts from real replay, stochastic
elite selection, and mixed real/model actor-critic updates. The upstream model predicts
reward plus observation delta and commonly refreshes every 250 environment steps.

LongiControl adaptations are explicit: the network predicts four vehicle/energy
quantities rather than the public observation and reward; track lookup, task costs,
elapsed time, metric history and termination are deterministic; rollout length is
frozen to one; the real fraction is a neutral 50%, because no audited MBPO setting is a
close match for this sensor-limited, low-dimensional, zero-tolerance constrained task;
and the downstream learner is frozen FSRL SACLag rather than upstream SAC. MBPO's 5%
real MuJoCo setting is therefore not imported as if it were task-independent.

## TD-MPC2

- Paper: *TD-MPC2: Scalable, Robust World Models for Continuous Control*, arXiv
  `2310.16828`.
- Official code: `https://github.com/nicklashansen/tdmpc2` at
  `e9f59321933cbc8e11a002b842adc7d4ffae8ff1`.

The inspected source combines an encoder, latent dynamics, reward and value/Q models,
a policy prior, temporal-difference learning and online trajectory optimization. The
repository's current episodic path is explicitly configured and is not its historical
default. TD-MPC2 is a useful modern continuous-control reference, but it is not the
selected design: using its latent objective and MPC action selection would change both
the learner and the final controller and would destroy the matched model-source
comparison.

## DreamerV3

- Paper: *Mastering Diverse Domains through World Models*, arXiv `2301.04104`.
- Official code: `https://github.com/danijar/dreamerv3` at
  `e01491fad6434b2245a3b8ca201dd7faedcc458c`.

The inspected implementation uses a recurrent state-space model and trains actor/value
components over multi-step latent imagination. It is contextual literature only. Its
representation, losses and learner are intentionally not introduced here.

## Frozen constrained learner

FSRL is pinned to `e056fc9498d5d037869533da7cf976acf462f918` with Tianshou
`0.5.1`. Inspection confirmed that `SACLagrangian.process_fn` constructs n-step targets
through replay-buffer indices, while `pre_update_fn` updates PID multipliers from the
episodic `stats_train["cost"]`. This has two consequences:

1. only completed real episodes call `pre_update_fn`;
2. a one-step counterfactual cannot legitimately acquire a second synthetic successor.

Real mixed-batch samples therefore retain frozen two-step processing. Synthetic H=1
samples use a one-step bootstrapped target. The model-disabled parity path calls the
unmodified FSRL `policy.update` and retains two-step processing for every sample. This
is the only unavoidable learner-interface adaptation and means frozen V2 is a close
contextual comparator, while Learned versus Physics remains the strict causal contrast.
