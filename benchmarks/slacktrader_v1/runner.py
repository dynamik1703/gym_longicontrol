"""SlackTrader wiring over the corrected exact-budget Hybrid SMDP runner."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from benchmarks.hybrid_rl_execution.runner import (
    BATCH_SIZE,
    GRADIENT_UPDATE_BUDGET,
    NATIVE_BUDGET,
    REPLAY_CAPACITY,
    TRAINING_SEEDS,
    WARMUP_NATIVE_TRANSITIONS,
    HybridRunner,
    UpdateBudgetAccumulator,
    cumulative_update_target,
)

from .agent import SlackTraderSAC
from .controller import PriceMPCController


class SlackTraderRunner(HybridRunner):
    """Use the frozen budget/accounting machinery with the price-MPC plant."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # No environment transition is taken by HybridRunner.__init__.  Replacing
        # the type-compatible policy and controller here cannot alter a task.
        self.agent = SlackTraderSAC(self.training_seed)
        self.controller = PriceMPCController()
        self.controller.reset()

    @classmethod
    def restore(
        cls,
        path: str | Path,
        *,
        expected_sha256: str | None,
        configuration_sha256: str,
        source_hashes: Mapping[str, str],
        attempt_id: str,
    ) -> SlackTraderRunner:
        restored = super().restore(
            path,
            expected_sha256=expected_sha256,
            configuration_sha256=configuration_sha256,
            source_hashes=source_hashes,
            attempt_id=attempt_id,
        )
        state = restored.agent.state_dict()
        restored.agent = SlackTraderSAC(restored.training_seed)
        restored.agent.load_state_dict(state)
        return restored


__all__ = [
    "BATCH_SIZE",
    "GRADIENT_UPDATE_BUDGET",
    "NATIVE_BUDGET",
    "REPLAY_CAPACITY",
    "TRAINING_SEEDS",
    "WARMUP_NATIVE_TRANSITIONS",
    "SlackTraderRunner",
    "UpdateBudgetAccumulator",
    "cumulative_update_target",
]
