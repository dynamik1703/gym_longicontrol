from pathlib import Path

import pytest

from benchmarks.feedback_chunk_diagnostic_v2.ledger import (
    build_pair_ledger,
    mark_started,
    validate_pair_ledger,
)
from benchmarks.feedback_chunk_diagnostic_v2.utils import atomic_json, load_json


def _split():
    return {
        "panel": {
            "id": "CHUNK_RL_DIAGNOSTIC_V2",
            "accepted_tasks": [
                {
                    "numeric_seed": 11_400_000 + index,
                    "route_sha256": f"{index:064x}",
                    "full_task_sha256": f"{index + 100:064x}",
                }
                for index in range(36)
            ],
        }
    }


def test_schedule_is_exactly_13_by_36():
    ledger = build_pair_ledger(_split())
    assert ledger["pair_count"] == 468
    assert validate_pair_ledger(ledger) == {"NOT_STARTED": 468}


def test_started_is_persisted_and_cannot_repeat(tmp_path: Path):
    path = tmp_path / "ledger.json"
    ledger = build_pair_ledger(_split())
    atomic_json(path, ledger)
    pair_id = ledger["pairs"][0]["pair_id"]
    mark_started(path, pair_id=pair_id, ownership_token="token", started_at_utc="now")
    assert load_json(path)["pairs"][0]["state"] == "STARTED"
    with pytest.raises(RuntimeError, match="not fresh"):
        mark_started(
            path,
            pair_id=pair_id,
            ownership_token="token",
            started_at_utc="later",
        )
