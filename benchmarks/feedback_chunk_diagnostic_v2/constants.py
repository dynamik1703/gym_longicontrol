"""Frozen paths and identities for the runtime-isolated Diagnostic V2."""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANONICAL = HERE / "canonical.json"
RUNTIME_MANIFESTS = HERE / "runtime_manifests.json"
SPLIT_LEDGER = HERE / "split_ledger.json"
PAIR_LEDGER = HERE / "pair_ledger.json"
PREPARATION_STATUS = HERE / "preparation_status.json"
AUTHORIZATION = HERE / "authorization.json"
EXECUTION_MANIFEST = HERE / "execution_manifest.json"
RESULTS = HERE / "results.json"

RUN_ROOT = ROOT / "runs/feedback-chunk-diagnostic-v2"
PAIR_ROOT = RUN_ROOT / "pairs"

V1_WORKTREE = Path("/Users/romanliessner/gitlab/gym_longicontrol_feedback_chunk_rl")
H_WORKTREE = Path("/Users/romanliessner/gitlab/gym_longicontrol_hybrid_rl_execution")
V1_MANIFEST = V1_WORKTREE / "runs/feedback-chunk-rl-v1/manifest.json"
V1_PAIR_LEDGER = V1_WORKTREE / ("runs/feedback-chunk-rl-v1/diagnostic/pair-ledger.json")
H_MANIFEST = H_WORKTREE / "runs/hybrid-rl-execution-v1/manifest.json"

BASE_COMMIT = "39e7b3c5edb435af2540a912633a966222bf96be"
BASE_TREE = "bca7eb626eae696e0aeea970f62e69a539c10d0c"
EXPECTED_BRANCH = "research/feedback-chunk-diagnostic-v2"
VERIFIED_REMOTE = "gym_longicontrol_origin"
VERIFIED_REMOTE_URL = "https://github.com/dynamik1703/gym_longicontrol.git"

PANEL_ID = "CHUNK_RL_DIAGNOSTIC_V2"
CANDIDATE_START = 11_400_000
TASK_COUNT = 36

CONTROLLER_ORDER = (
    "CAUSAL60",
    "TIME_AWARE",
    "FIXED_PRICE_MPC_V2",
    "ADAPTIVE_PRICE_MPC_V2",
    "HYBRID_H11",
    "HYBRID_H29",
    "HYBRID_H47",
    "CHUNK3_S11_300K",
    "CHUNK3_S29_300K",
    "CHUNK3_S47_300K",
    "RESIDUAL1S_S11_300K",
    "RESIDUAL1S_S29_300K",
    "RESIDUAL1S_S47_300K",
)
NONLEARNING_CONTROLLERS = frozenset(CONTROLLER_ORDER[:4])
H_CONTROLLERS = frozenset(CONTROLLER_ORDER[4:7])
C_LEARNED_CONTROLLERS = frozenset(CONTROLLER_ORDER[7:])
C_CONTROLLERS = NONLEARNING_CONTROLLERS | C_LEARNED_CONTROLLERS

H_SEEDS = {"HYBRID_H11": 11, "HYBRID_H29": 29, "HYBRID_H47": 47}
C_IDENTITIES = {
    "CHUNK3_S11_300K": ("CHUNK3_RESIDUAL_SAC", 11),
    "CHUNK3_S29_300K": ("CHUNK3_RESIDUAL_SAC", 29),
    "CHUNK3_S47_300K": ("CHUNK3_RESIDUAL_SAC", 47),
    "RESIDUAL1S_S11_300K": ("RESIDUAL1S_SAC", 11),
    "RESIDUAL1S_S29_300K": ("RESIDUAL1S_SAC", 29),
    "RESIDUAL1S_S47_300K": ("RESIDUAL1S_SAC", 47),
}

V1_LEDGER_SHA256 = "38cc54368e7b8795af47192f3cadf9d1da0620a2ad7b5d453dfd7a8c1f48e8c2"
V1_MANIFEST_SHA256 = "332245ee1d155c74b08084178601eed262a8742617b8f652d6dbc2e2e01c01d7"
H_MANIFEST_SHA256 = "5272170534b026948a74d18bccc375cf36505b0d7c8d48c7e0fc9c97bd9f7679"
POWER_MODEL_SHA256 = "88a63fddfabf0f61123b3eee90fc84dba363911697d8e5cc10eb1685393d866d"

H_CHECKPOINTS = {
    "HYBRID_H11": {
        "sha256": "a16cd663a4cd046532490b74a414e37acee9b37ca0c0b886753fa8d767f4f7ed",
        "attempt_id": "5f5801d9-d98c-4ace-9b1a-34476759f60a",
    },
    "HYBRID_H29": {
        "sha256": "b4c6f5f484cc4a68b044b39f70db8d5f8245098729b3bb06749dbb46bdb9af55",
        "attempt_id": "97a40dfa-c8ed-4334-b2c3-0351e71697a8",
    },
    "HYBRID_H47": {
        "sha256": "3a67097e07360a07c8d5e5a276cf388434476943941085aea98b4a85254d82cf",
        "attempt_id": "4553b298-5797-493f-88fe-075de3786dff",
    },
}
C_CHECKPOINTS = {
    "CHUNK3_S11_300K": {
        "sha256": "de59e6435e93f81b08931f414b8615226c424e025c44090e31d107d2e61770e0",
        "attempt_id": "ba67e42e-5372-4f47-943e-75f4a8c84516",
    },
    "CHUNK3_S29_300K": {
        "sha256": "7b71c9c2206e0e3a5e96486812a8a85925b4ceef73bc7bdfce7a00482410939d",
        "attempt_id": "bb5b2a79-46bf-44dc-bdb5-715081b168da",
    },
    "CHUNK3_S47_300K": {
        "sha256": "207a5cc0feae49bab0a2027fcf1ab8432a073c7776a468f7d1af5f2d73ed813f",
        "attempt_id": "c2d827a4-0214-4a5e-9abb-4a7cb4cc8970",
    },
    "RESIDUAL1S_S11_300K": {
        "sha256": "90a832a9839a6a7f22298c91e45d61d8e2d3f47a508dcd6f4a1475cd9a056591",
        "attempt_id": "86d9b5cb-7482-4d1b-bcbd-ea83aa90a028",
    },
    "RESIDUAL1S_S29_300K": {
        "sha256": "46e5aa9ce633a88dbca012db352f1d923daf68de7ced29c2c0a2aa1d8167bead",
        "attempt_id": "09f0cac3-f2db-4918-8d22-b28a26f7fe5a",
    },
    "RESIDUAL1S_S47_300K": {
        "sha256": "282544146f0ece5a6b37b9032907149fe808100d5955bb933129538ecd363424",
        "attempt_id": "2e5ad03c-6ce5-4bba-9001-284625ecaf04",
    },
}

ABSOLUTE_FLOAT_TOLERANCE = 1.0e-12
PAIR_COUNT = len(CONTROLLER_ORDER) * TASK_COUNT
