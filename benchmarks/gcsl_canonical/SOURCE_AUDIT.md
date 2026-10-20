# Primary-source audit

This audit was completed before any canonical GCSL policy training and without
using policy-performance pilots.

## Pinned sources

- Paper: Ghosh et al., *Learning to Reach Goals via Iterated Supervised
  Learning*, arXiv `1912.06088v4` (2 October 2020), published at ICLR 2021 as
  OpenReview `rALA0Xo6yNJ`.
- Official code: `dibyaghosh/gcsl` commit
  `cfae5609cee79e5a2228fb7653451023c41a64cb`, tree
  `b5672e00ed73787e85c193fab79f1aa51edbd9c3` (17 September 2020).
- File hashes are recorded in `upstream.json`.
- The immutable prior LongiControl port is commit
  `132fa370d60820d71ef5d8e3a988be98f7defa7c` on `research/gcsl`.

The pinned repository has no top-level `LICENSE`, `COPYING`, or `NOTICE`. A
license inside the ROBEL dependency does not license the GCSL repository. This
study therefore vendors no upstream source and independently implements the
audited semantics.

## Audited semantics

The paper samples a rollout goal, executes a goal-conditioned policy, relabels
a source action with a later achieved state and lag, and maximizes
`log pi(a | s, g, h)`. There is no critic, Q-function, Bellman target, reward
regression, cost signal, or negative-goal set.

The release stores trajectories and samples same-trajectory strict-future
pairs on demand. Its default principal policy does not receive the horizon;
horizon conditioning is an ablation. Adam uses learning rate `5e-4` and batch
size 256. Evaluation is greedy.

The official release does not define a continuous stochastic likelihood. Box
actions are discretized to categorical actions. LongiControl retains its native
scalar action in `[-1, 1]`; the independently implemented tanh-transformed
Gaussian likelihood is therefore an explicit continuous-action adaptation.

## Frozen LongiControl port

The canonical study uses the 12D state (eight sensor values plus normalized
position, previous position, elapsed time, and accumulated maximum speed
violation), the exact 3D projected-future goal, and canonical command
`[1, 1, 1]`. Replay draws a same-episode strict physical future with probability
proportional to `0.99**lag`. This deliberately matches the registered
projected-goal CRL sampler rather than the upstream two-uniform-index sampler.

Collection and deterministic future evaluation query only the canonical goal;
updates use only achieved projected futures. The cold-start extrapolation risk
is disclosed: replay never inserts demonstrations or fabricates success.
