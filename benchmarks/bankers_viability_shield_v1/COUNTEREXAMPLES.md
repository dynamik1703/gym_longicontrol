# Falsification record

The proposed invariant is tested by synthetic edge-of-mask rollouts, exact brake-policy execution, boundary fixtures and an offline state/action search. The search uses the full synthetic route only in its independent verifier. Online inputs remain causal. Counts and any concrete witness are retained in counterexample_search.json and native evidence.

A separate negative control searches the unsound shortcut that gives a successor a fresh 150 m without a new observation. A witnessed failure of that shortcut is not a failure of the persistent pre-action-envelope invariant, nor a demonstrated failure of the inherited complete controller inside its 60 km/h proposal domain. Its exact state and physical violating endpoint are preserved if found.

If an admitted trajectory violates the proposed invariant, retain its complete prefix and certificate and stop. Do not redefine the invariant or relax the zero speed tolerance to conceal it. Empirical absence of counterexamples is not the mathematical proof.

Observed result: 6720 proposals at 320 synthetic states, 6470 admitted offline
braking witnesses and inductive successor checks, zero core counterexamples.
A separate latency shortcut counterexample starts at 25.6625 m/s, proposes +1,
and crosses the unseen sign at 150.000001 m with endpoint velocity
5.642955973934301 m/s at position 150.02593711687808 m. Its excess is
0.08740041837874557 m/s even though successor braking distance plus 0.5 m is
below 150 m. BankerShield rejects it using the old absolute frontier.
This example is outside the inherited controller's 60 km/h proposal domain.
It is retained as a negative control, not a core-invariant falsification or a
claim that an inherited admitted trajectory from reset is unsafe.
