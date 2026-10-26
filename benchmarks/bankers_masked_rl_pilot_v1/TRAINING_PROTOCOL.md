# Planned matrix; scientific execution stopped

Exactly RAW_DISCRETE_SAC, BANKER_POST_SAC and BANKER_PRE_SAC, seed 11, were
planned at 50,000 native transitions and 5,000 optimizer updates each. CANONICAL_11
contains 865 ordered tasks and hashes to
`08e8e5b0f03dd7caae95b3f1175edaa22a29608cbc9a554fea6e3be3e24cfe3c`.
All arms were to use the same task order, with different policies allowed to
finish episodes at different transition counts.

The integer cumulative update rule is
`max(0,min(5000,((N-10000)*5000)//40000))`. The first update is at 10,008 and
the last at 50,000; no warm-up update burst occurs. The only primary endpoint
is 50k. No best-checkpoint selection or tuning on Diagnostic outcomes is allowed.

Before scientific reset: all A–P gates must pass, a complete measured <=8h and
<=2GiB projection must pass, and source/protocol/config must be committed.
Gate N failed. No scientific Attempt was STARTED, no training reset occurred,
and no complete execution transaction was implemented or released.
The component training module must not be called outside synthetic tests in
this stopped revision. The public execution entry point fails closed.
