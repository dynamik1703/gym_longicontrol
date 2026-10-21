# Study Design

At each 1 Hz decision boundary, FixedPrice, AdaptivePrice or SlackTrader emits
one normalized scalar. A frozen logarithmic map converts it to `lambda` in
kWh/s. The shared causal planner solves a 100 m spatial problem, returns the
first ten native actions, and the unchanged physical gates revalidate each
action at 10 Hz. Newly visible transitions trigger causal replanning.

```text
8D observation + causal memory -> 16D strategy observation
                                  |
                    fixed / adaptive / learned scalar
                                  |
                    logarithmic lambda mapping
                                  |
                     shared 100 m MPC planner
                                  |
                 shared native physical safety gates
                                  |
                       StochasticTrack-v1
```

The three primary price methods differ only in scalar selection. They share the
same candidate lattice, exact dynamics, terminal treatment, causal memory,
action interface, fallback, safety gates and decision frequency.

The proposed learner is the corrected one-scalar constrained SAC from Hybrid:
two 256-unit actor layers, twin reward critics, explicit speed and terminal
cost critics, and a separate native-action intervention regularizer. Replay
stores proposed normalized price, never executed actuator action.

The frozen future budget is 300,000 real native transitions and 30,000 updates
per seed for seeds 11, 29 and 47, with no best-checkpoint selection. It is a
design artifact only: main training is not authorized.
