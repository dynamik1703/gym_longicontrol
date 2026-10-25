# Execution integrity

The master is the only ledger writer. For each pair it performs:

1. worker `PREPARE` with a frozen untouched identity;
2. durable master `STARTED` plus an ownership-token hash;
3. worker revalidation of `STARTED`, then the first environment reset;
4. independent in-memory verification;
5. atomic compressed evidence and result writes plus hash readback;
6. durable `COMPLETED`.

Completed pairs cannot repeat. A crash or error after `STARTED` leaves that pair
open, records an integrity stop, and cannot be recovered without a separately
approved full-native-state protocol. V1 ledgers are read-only and checked
byte-for-byte. Validation and Paper episode counters must remain zero.
