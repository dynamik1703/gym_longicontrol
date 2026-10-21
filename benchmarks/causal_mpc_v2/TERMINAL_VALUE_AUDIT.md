# Terminal Value Audit

V1 charged every slow horizon endpoint for accelerating back to a common speed
near 60 km/h. That is causal but can systematically prefer fast endpoints even
when the route deadline requires only a much lower continuation speed.

V2 instead computes a causal deadline-paced continuation target from remaining
distance and remaining usable time, adds the frozen 5 km/h planning margin, and
clips it to the generator minimum, 60 km/h causal cap and causally known limit.
It uses exact dynamics to connect terminal velocity to that target: acceleration
when below, coasting when above, and mandatory braking for a known lower limit.
Regenerative energy in this synthetic connection is clipped at zero so a high
terminal speed cannot receive an artificial horizon-edge credit. Remaining
distance is valued at the exact production-model power required to hold the
continuation target, with remembered visible lower limits incorporated.

The continuation returns both estimated energy and time. There is no overlap
with horizon distance, so neither component is double counted. A state that
cannot complete the assumed causal continuation by the deadline is rejected.

This is a deterministic heuristic estimate under an explicit constant-speed
unknown-tail assumption. It is neither a lower/upper bound nor an exact value
function. It does not use the full route and it neither assigns zero future cost
to arbitrary slow states nor demands acceleration to 60 km/h. Deterministic
bias fixtures compare it with V1 and with direct exact-dynamics connections.

