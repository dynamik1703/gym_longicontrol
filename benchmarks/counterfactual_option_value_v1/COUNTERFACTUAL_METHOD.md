# Counterfactual Method

Snapshots contain the environment and its RNG, observation, full causal
controller and fallback state, persistent option memory/history, held price,
strategic clock, pending values and global RNG states. Hidden route content may
exist only inside the simulator snapshot needed for exact restoration.

The reference makes its normal frozen decision. An alternative overrides exactly
one first high-level START/CONTINUE decision and then returns to the same frozen
controller while preserving the new option memory. Every continuation runs to
termination or the unchanged 1800-step outer limit. Full remaining-episode signed
energy and travel time—not one-second proxies—form the paired outcome.

