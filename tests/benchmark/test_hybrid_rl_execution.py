from __future__ import annotations

import json
from pathlib import Path

from benchmarks.hybrid_rl_execution.evaluation import (
    verify_frozen_splits_without_opening,
)
from benchmarks.hybrid_rl_execution.execution import require_main_training_authorized
from benchmarks.hybrid_rl_execution.macro_transition import GAMMA_DECISION, GAMMA_NATIVE


def test_hybrid_execution_is_smdp_ready_but_training_fail_closed():
    assert abs(GAMMA_NATIVE**10 - GAMMA_DECISION) < 1e-15
    splits = verify_frozen_splits_without_opening()
    assert splits["diagnostic"]["count"] == 18
    assert splits["validation"]["performance_episodes"] == 0
    try:
        require_main_training_authorized()
    except PermissionError:
        pass
    else:
        raise AssertionError("main training unexpectedly authorized")
    status_path = Path("benchmarks/hybrid_rl_execution/preparation_status.json")
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        assert status["protocol_revision"] == 3
        assert status["ready_for_main_training"]
        assert not status["main_training_started"]
        assert status["validation"]["performance_episodes"] == 0
