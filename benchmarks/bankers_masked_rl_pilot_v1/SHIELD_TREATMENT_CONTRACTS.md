# Treatment contracts

All learners propose indices of the exact V4 grid [-1,-0.9,...,0.9,1]. The
continuous-action production environment is unchanged.

RAW samples all 21 and executes directly; observed violations remain outcomes.
POST samples all 21, computes the frozen V4 safe grid, and replaces an invalid
proposal by the nearest certified value, breaking ties toward the lower value.
Replay retains the original proposal. PRE computes the mask before sampling,
samples only certified actions and performs no normal post-action replacement.

Both Banker arms use `CompactShield` around the unchanged V3 kinematic kernel
and the complete original `BankersShield` certificate immediately before
execution. A discrepancy stops execution. No backup, envelope, frontier,
native timestep, margin, grid or exact-speed criterion was changed.

These are implemented adapter intentions. Independent native verification
did not pass, so no scientific training safety claim is made.
