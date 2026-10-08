# Primary-source audit

This audit was completed before any GCSL policy training and without consulting
the running projected-goal CRL results.

## Pinned sources

- Paper: Ghosh et al., *Learning to Reach Goals via Iterated Supervised
  Learning*, arXiv `1912.06088v4` (2 October 2020), published at ICLR 2021 as
  OpenReview `rALA0Xo6yNJ`.
- Official code: `dibyaghosh/gcsl` commit
  `cfae5609cee79e5a2228fb7653451023c41a64cb`, tree
  `b5672e00ed73787e85c193fab79f1aa51edbd9c3` (17 September 2020).
- File hashes are recorded in `upstream.json`.

The pinned repository has no top-level `LICENSE`, `COPYING`, or `NOTICE`. A
license inside the vendored ROBEL dependency applies to that dependency, not to
the GCSL repository. Consequently there is no explicit upstream permission
grant to reproduce the code. This study vendors no upstream source and
implements the audited semantics independently.

The upstream environment pins Python 3.5.2, NumPy 1.11.3, Gym 0.10.5 and
PyTorch 1.1.0 plus old experiment infrastructure. None was installed into the
user's base environment.

## Source-faithful GCSL

The paper's algorithm repeatedly samples a rollout goal, executes the current
goal-conditioned policy, relabels a source action with a later achieved state
and its lag, then maximizes `log pi(a | s, g, h)` over replay. It has no critic,
Bellman target, value function, reward regression, or negative goal set.

The release implements this in `gcsl/algo/gcsl.py` and
`gcsl/algo/buffer.py`:

- A trajectory is stored as states and actions. Sampling chooses one trajectory
  uniformly, draws one index uniformly from `[0, L-2]` and another from
  `[0, L-1]`, increments equality, and orders the two. The smaller index is the
  source and the larger is the goal. This is on-demand same-trajectory strict
  future relabeling; it is not a uniform draw over all ordered `(t, t+k)` pairs.
- The returned lag is encoded as a length-`T` reverse-temperature Boolean
  vector: element `j` is `1[j >= k]`.
- Algorithm 1 in the paper conceptually adds all `T choose 2` tuples. The code
  instead samples equivalent tuple types from a bounded replay of 20,000 fixed
  length trajectories.
- Collection samples `env.sample_goal()`. The paper's principal task goals are
  uniform over reachable configurations. The first 10,000 transitions use
  maximum discrete exploration noise; the released default thereafter uses a
  greedy policy and zero exploration noise.
- The default policy is a two-hidden-layer `[400, 300]` ReLU network. The goal
  is concatenated once with state. Although horizon-aware network support
  exists, `variants.default_markov_policy` sets `max_horizon=None` with the
  comment “Do not pass in horizon.” The paper likewise says the principal
  experiments work without horizon; remaining-horizon conditioning is an
  ablation (“Time-Varying Policy”).
- Adam uses learning rate `5e-4`, batch size 256 and one gradient update per
  environment step. Updates are grouped after each trajectory (default length
  50). A 20% trajectory validation buffer is excluded from training.
- Evaluation is greedy, samples target goals from the environment, and measures
  final goal distance/success.

## Continuous actions in the official source

The official release does **not** define a continuous stochastic likelihood.
`variants.discretize_environment` wraps every Box action space in
`DiscretizedActionEnv`, with three values per dimension by default. The policy
uses categorical cross-entropy (or independent categorical cross-entropies for
large product action spaces). Describing the released code as Gaussian NLL or
MSE would therefore be incorrect.

## LongiControl adaptation

LongiControl retains its native scalar action in `[-1, 1]`. Discretizing it to
three commands would gratuitously alter control authority. The smallest
defensible stochastic adaptation is a diagonal Gaussian in pre-tanh space with
a tanh transform. It preserves maximum-likelihood training and bounded sampled
actions, but is explicitly an adaptation rather than upstream action semantics.

The principal policy omits horizon conditioning because the audited default
does. Absolute elapsed time remains a state value, while deadline satisfaction
is a projected goal bit. Neither is a future lag. The replay still records lag
for diagnostics.

## Deliberate match to projected-goal CRL

The learner comparison freezes the already-preregistered CRL strict-future
distribution: one same-episode future with probability proportional to
`0.99**lag`. Transition rows explicitly distinguish the pre-action source from
their strict post-action outcome, so physical lag one is included. This differs
from the official two-uniform-index sampler, but does not invalidate direct
hindsight maximum likelihood. It holds the experienced future distribution
fixed across learners, preserves full multi-horizon relabeling, and prevents a
sampler difference from driving the comparison.

Collection and deterministic evaluation always query `[1, 1, 1]`; supervised
updates use only the projected goal of an actually recorded future. This
degenerate rollout-goal distribution differs from upstream's broad reachable
goal distribution and creates a documented cold-start extrapolation risk. It
is nevertheless methodologically coherent: replay never fabricates success,
and iterative self-supervision still operates on every outcome the policy
actually reaches.
