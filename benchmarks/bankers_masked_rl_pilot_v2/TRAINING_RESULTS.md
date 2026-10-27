# Training results

|Method|Native transitions|Updates|Status|
|---|---:|---:|---|
|RAW_DISCRETE_SAC|0/50000|0/5000|NOT_STARTED|
|BANKER_POST_SAC|0/50000|0/5000|NOT_STARTED|
|BANKER_PRE_SAC|0/50000|0/5000|NOT_STARTED|

Completed-run records retain actor/checkpoint hashes, all optimizer diagnostics,
partial episode state and actual task exposure. Different episode lengths may
expose different prefixes of the identical canonical task order.
