# Falsification record

The proposed invariant is tested by synthetic edge-of-mask rollouts, exact brake-policy execution, boundary fixtures and an offline state/action search. The search uses the full synthetic route only in its independent verifier. Online inputs remain causal. Counts and any concrete witness are retained in counterexample_search.json and native evidence.

A separate negative control searches the unsound shortcut that gives a successor a fresh 150 m without a new observation. A witnessed failure of that shortcut is not a failure of the persistent pre-action-envelope invariant, nor a demonstrated failure of the inherited complete controller inside its 60 km/h proposal domain. Its exact state and physical violating endpoint are preserved if found.

If an admitted trajectory violates the proposed invariant, retain its complete prefix and certificate and stop. Do not redefine the invariant or relax the zero speed tolerance to conceal it. Empirical absence of counterexamples is not the mathematical proof.
