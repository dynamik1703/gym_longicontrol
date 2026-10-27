# Independent causal predecessor reconstruction

V2 owns an auditor CausalMemory starting at the known reset state. At every
current observation it preserves a deep copy before ingest, reconstructs the
ordinary eight-vector from its propagated state and the actual audit-only track,
ingests that vector, and calls verify_envelope(current, previous, track).
The two memories are distinct and previous.last_observation is null. The frozen
helper checks lower bounds against physical route limits at discontinuities and
adjacent binary64 points and checks monotone retained information.

Only after these checks does V2 compare the complete producer snapshot, binding,
envelope and remembered signs. It reconstructs the selected action and complete
physical successor, energy, exact excess, reward/costs and lifecycle. The auditor's
authoritative memory advances by the executed action and remains before the next
ingest; a separate clone verifies the recorded successor observation and mask
binding. No producer snapshot is restored as authoritative audit state.

At actual reset, task identity, cursor and zero episode-local counter are checked;
memory and speed-metric history reset without losing the global transition count.
The track is copied into auditor ownership and an in-episode track substitution
is rejected. Track/route/seed objects never enter actor or shield inference.
