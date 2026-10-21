"""Preregistered causal selection of exactly two non-reference alternatives."""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.temporal_options_v1.option_selector import OptionCandidate


@dataclass(frozen=True)
class AlternativeSet:
    reference_action_id: str
    alternative_action_ids: tuple[str, ...]
    valid_action_ids: tuple[str, ...]


def _start_rank(candidate: OptionCandidate) -> tuple[float, float, float, str]:
    return (
        float(candidate.objective_kwh),
        float(candidate.future_option_time_s + candidate.terminal_time_s),
        float(
            candidate.future_option_energy_kwh
            + candidate.terminal_energy_kwh
            + candidate.lateness_penalty_kwh
        ),
        candidate.profile_id,
    )


def select_alternatives(
    *, reference_action_id: str, candidates: tuple[OptionCandidate, ...]
) -> AlternativeSet:
    valid = tuple(candidate for candidate in candidates if candidate.feasible)
    valid_ids = tuple(candidate.high_level_action for candidate in valid)
    if reference_action_id not in valid_ids:
        raise ValueError("reference action is not valid")
    selected: list[str] = []
    if "CONTINUE" in valid_ids and reference_action_id != "CONTINUE":
        selected.append("CONTINUE")
    starts = sorted(
        (
            candidate
            for candidate in valid
            if candidate.high_level_action.startswith("START(")
            and candidate.high_level_action != reference_action_id
        ),
        key=_start_rank,
    )
    for candidate in starts:
        if len(selected) >= 2:
            break
        if candidate.high_level_action not in selected:
            selected.append(candidate.high_level_action)
    return AlternativeSet(
        reference_action_id=reference_action_id,
        alternative_action_ids=tuple(selected[:2]),
        valid_action_ids=valid_ids,
    )
