"""Frozen identities for Hybrid RL Validation V2 preparation."""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL_WORKTREE = Path(
    "/Users/romanliessner/gitlab/gym_longicontrol_hybrid_rl_execution"
)

V1_ID = "HYBRID_RL_VALIDATION_V1"
V1_STATE = "INCIDENT_ABORTED"
V1_ORDERED_SHA256 = (
    "f3864fed8725874dc315004375af0d2850dc2d2d56957d52c64d043ca6b3352b"
)
V2_ID = "HYBRID_RL_VALIDATION_V2"
V2_STATE = "FROZEN_UNOPENED"
V2_CANDIDATE_START = 10_200_000
V2_TASK_COUNT = 36

ORIGINAL_MAIN_TRAINING_COMMIT = (
    "7676f2c77b82fd8273e8c38722082ffb7afd5c60"
)
INCIDENT_PRESERVATION_COMMIT = (
    "27cff19dadebaafbcd905fd3a2ecbe4ac8f0eeaa"
)
TRAINING_EXECUTION_CONFIGURATION_SHA256 = (
    "73e2bd6d960504eff120a37aaba60f9c3e17cb353408a67f7feddb97bf33253e"
)

CONTROLLER_ORDER = ("CAUSAL60", "TIME_AWARE", "H11", "H29", "H47")
POLICY_SEEDS = {"H11": 11, "H29": 29, "H47": 47}
EXPECTED_CHECKPOINT_SHA256 = {
    "H11": "a16cd663a4cd046532490b74a414e37acee9b37ca0c0b886753fa8d767f4f7ed",
    "H29": "b4c6f5f484cc4a68b044b39f70db8d5f8245098729b3bb06749dbb46bdb9af55",
    "H47": "3a67097e07360a07c8d5e5a276cf388434476943941085aea98b4a85254d82cf",
}
EXPECTED_ACTOR_SHA256 = {
    "H11": "2f574832349948dae731b57e4e85dca5ee797665865c5f332e7a7feb3612488e",
    "H29": "bd1de85fc0fe586e39831f2dea917fa5dd131530a625a58dd393f1e2094ace65",
    "H47": "135486355eb36def746f38ef20638a0fdd9aab02370e392a108cda66f87dfad6",
}

SPLIT_LEDGER = HERE / "split_ledger.json"
PAIR_LEDGER = HERE / "pair_ledger.json"
AUTHORIZATION = HERE / "authorization.json"
PREPARATION_STATUS = HERE / "preparation_status.json"
