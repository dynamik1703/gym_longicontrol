from __future__ import annotations

from pathlib import Path

from benchmarks.feedback_chunk_rl_v1.training import TrainingRunner


def _runner(attempt_id: str) -> TrainingRunner:
    return TrainingRunner(
        method="CHUNK3_RESIDUAL_SAC",
        training_seed=11,
        attempt_id=attempt_id,
        configuration_sha256="recovery-fixture-configuration",
        source_hashes={"fixture": "source"},
        native_budget=80,
        update_budget=0,
        warmup_native_transitions=79,
        batch_size=2,
        replay_capacity=128,
    )


def _assert_recovery_at_step(path: Path, checkpoint_step: int) -> None:
    runner = _runner(f"attempt-{checkpoint_step}")
    restored: TrainingRunner | None = None
    try:
        while runner.native_transition_count < checkpoint_step:
            runner.advance_native()
        assert runner.controller.active_chunk is not None
        assert runner.controller.active_chunk.phase(runner.controller.native_step) in (
            1,
            2,
        )
        metadata = runner.save(path)
        expected = [runner.advance_native() for _ in range(14)]
        restored = TrainingRunner.restore(
            path,
            expected_sha256=metadata["sha256"],
            configuration_sha256="recovery-fixture-configuration",
            source_hashes={"fixture": "source"},
            attempt_id=f"attempt-{checkpoint_step}",
        )
        observed = [restored.advance_native() for _ in range(14)]
        assert observed == expected
        assert restored.native_transition_count == runner.native_transition_count
        assert restored.macro_transition_count == runner.macro_transition_count
        assert restored.gradient_update_count == runner.gradient_update_count
        assert (
            restored.replay.state_dict()["position"]
            == runner.replay.state_dict()["position"]
        )
    finally:
        runner.close()
        if restored is not None:
            restored.close()


def test_exact_recovery_inside_second_two(tmp_path: Path) -> None:
    _assert_recovery_at_step(tmp_path / "second-two.pkl", checkpoint_step=12)


def test_exact_recovery_inside_second_three(tmp_path: Path) -> None:
    _assert_recovery_at_step(tmp_path / "second-three.pkl", checkpoint_step=22)
