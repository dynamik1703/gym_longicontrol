# Controlled seed-11 training protocol

Exactly RAW_DISCRETE_SAC, BANKER_POST_SAC and BANKER_PRE_SAC each receive seed 11,
50,000 native transitions and 5,000 gradient updates. All use the same 865-row
CANONICAL_11 ordered metadata with recomputed SHA-256
08e8e5b0f03dd7caae95b3f1175edaa22a29608cbc9a554fea6e3be3e24cfe3c.
Different episode durations may consume different prefix lengths; disclose them.

Cumulative updates=max(0,min(5000,((N-10000)*5000)//40000)). No updates through 10k;
first at 10008, final at 50000. No update burst, no action repeat, no best checkpoint.
Final 50k is the sole primary endpoint. Native-budget cutoff is not task failure.

Before any scientific reset: engineering tests pass, the complete resource model
passes <=8h and <=2GiB, scientific source/config/protocols are committed, and the
exactly-once sealed preflight passes. Durable STARTED precedes reset. Immutable
V2 Attempt IDs are unique by treatment/seed. Exclusive execution ownership applies.
Runtime source checks occur every 1000 transitions. Full final checkpoint, actor
hash, independent disk audit, all replay-slot comparisons and completion seal
are required. No Diagnostic opening until all three endpoints are sealed.
No reward, architecture, entropy, safety or budget tuning based on outcomes.
No full-scale 300k, final Validation or Paper execution is authorized.
