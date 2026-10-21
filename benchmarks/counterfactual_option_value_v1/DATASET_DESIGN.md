# Dataset Design

`CF_OPTION_FIT_V1`, `CF_OPTION_CALIBRATION_V1` and `CF_OPTION_TEST_V1` are
independently selected route-clean panels. All snapshots from a route stay in its
panel. There is no row-level split. Ordered hashes, all scanned candidates and
rejection reasons are stored in `split_ledger.json` before policy evaluation.

Test becomes opened study data after scoring and can never be described later as
an untouched final RL Validation panel. Paper seeds 4000–4017 are excluded only
numerically; Paper content is never constructed or fingerprinted.

