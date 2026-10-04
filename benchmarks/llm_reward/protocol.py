"""Immutable configuration for the preregistered LLM reward search."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from benchmarks.scalar_sb3.config import SACConfiguration
from gym_longicontrol.domain.task import TaskSpecification

DEFAULT_PROTOCOL_PATH = Path(__file__).with_name("search_protocol.json")


def _integer_tuple(name: str, raw: Any) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{name} must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
        raise ValueError(f"{name} must contain integers")
    values = tuple(raw)
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must not contain duplicates")
    return values


@dataclass(frozen=True)
class SearchBudget:
    candidate_counts_by_generation: tuple[int, ...]
    maximum_candidates: int
    maximum_generation: int
    parent_count: int
    generation_zero_minimum_distinct_functional_forms: int

    def __post_init__(self):
        if self.candidate_counts_by_generation != (5, 3, 3):
            raise ValueError("The frozen generation budget must be 5, 3, 3")
        if self.maximum_candidates != 11 or self.maximum_generation != 2:
            raise ValueError("The frozen search must stop after 11 candidates / G2")
        if self.parent_count != 2:
            raise ValueError("Exactly two cumulative leaders become parents")
        if self.generation_zero_minimum_distinct_functional_forms != 3:
            raise ValueError("Generation 0 requires at least three functional forms")


@dataclass(frozen=True)
class ScreeningProtocol:
    training_seed: int
    simulator_transitions_per_candidate: int
    evaluation_checkpoints: tuple[int, ...]
    development_split_id: str
    development_tracks: tuple[int, ...]

    def __post_init__(self):
        if self.training_seed != 11:
            raise ValueError("The screening seed is frozen to 11")
        if self.simulator_transitions_per_candidate != 50_000:
            raise ValueError("Every screening run must receive exactly 50k steps")
        if self.evaluation_checkpoints != (10_000, 25_000, 50_000):
            raise ValueError("Screening checkpoints must be 10k, 25k and 50k")
        if self.development_tracks != tuple(range(2000, 2009)):
            raise ValueError("Reward search is restricted to Development 2000--2008")
        if self.development_split_id != "development-2000-2008-v1":
            raise ValueError("Unexpected Development split identifier")


@dataclass(frozen=True)
class TrackPolicy:
    historical_exploratory: tuple[int, ...]
    validation: tuple[int, ...]
    paper_final_test_reserved: tuple[int, ...]

    def __post_init__(self):
        if self.historical_exploratory != tuple(range(1000, 1009)):
            raise ValueError("Historical exploratory split changed")
        if self.validation != tuple(range(3000, 3009)):
            raise ValueError("Validation split changed")
        if self.paper_final_test_reserved != tuple(range(4000, 4018)):
            raise ValueError("Paper-final split changed")
        splits = (
            set(self.historical_exploratory),
            set(self.validation),
            set(self.paper_final_test_reserved),
        )
        if any(
            left & right
            for index, left in enumerate(splits)
            for right in splits[index + 1 :]
        ):
            raise ValueError("Track splits must be disjoint")


@dataclass(frozen=True)
class RetryPolicy:
    maximum_repair_attempts_per_candidate: int
    repair_feedback: str
    replacement_after_technical_failure: bool

    def __post_init__(self):
        if self.maximum_repair_attempts_per_candidate != 1:
            raise ValueError("Exactly one repair attempt is allowed")
        if self.repair_feedback != "validation-errors-only":
            raise ValueError("Repair feedback must contain validation errors only")
        if self.replacement_after_technical_failure:
            raise ValueError("A technical failure must consume its candidate slot")


@dataclass(frozen=True)
class RankingProtocol:
    order: tuple[str, ...]
    substantial_progress_threshold: float
    energy_scope: str

    def __post_init__(self):
        expected = (
            "development_rsr_desc",
            "completion_rate_desc",
            "median_incomplete_route_progress_desc",
            "deadline_compliance_among_completed_desc",
            "speed_compliance_among_relevant_desc",
            "speed_violation_severity_among_relevant_asc",
            "feasible_energy_kwh_asc",
            "candidate_id_asc",
        )
        if self.order != expected:
            raise ValueError("Lexicographic ranking order changed")
        if self.substantial_progress_threshold != 0.5:
            raise ValueError("Substantial route progress is frozen to 50 percent")
        if self.energy_scope != "feasible-episodes-only":
            raise ValueError("Energy may rank only feasible behavior")


@dataclass(frozen=True)
class FinalEvaluationProtocol:
    training_seeds: tuple[int, ...]
    simulator_transitions_per_seed: int
    evaluation_split_id: str
    validation_tracks: tuple[int, ...]
    requires_closed_search: bool
    winner_only: bool

    def __post_init__(self):
        if self.training_seeds != (11, 29, 47):
            raise ValueError("Final training seeds changed")
        if self.simulator_transitions_per_seed != 300_000:
            raise ValueError("Final training budget must be 300k per seed")
        if self.validation_tracks != tuple(range(3000, 3009)):
            raise ValueError("Final evaluation must use Validation 3000--3008")
        if self.evaluation_split_id != "validation-3000-3008-v1":
            raise ValueError("Unexpected final Validation split identifier")
        if not self.requires_closed_search or not self.winner_only:
            raise ValueError("Validation requires a closed search and its sole winner")


@dataclass(frozen=True)
class SearchProtocol:
    schema_version: int
    protocol_id: str
    phase: str
    environment_id: str
    max_episode_steps: int
    task: TaskSpecification
    search_budget: SearchBudget
    screening: ScreeningProtocol
    sac: SACConfiguration
    track_policy: TrackPolicy
    retry_policy: RetryPolicy
    ranking: RankingProtocol
    final_evaluation: FinalEvaluationProtocol
    allowed_candidate_imports: tuple[str, ...]
    information_barrier: tuple[str, ...]

    def __post_init__(self):
        if self.schema_version != 1 or self.protocol_id != "llm-reward-search-v1":
            raise ValueError("Unsupported LLM reward protocol")
        if self.phase != "phase-1-infrastructure-frozen":
            raise ValueError("Protocol phase must remain frozen before search")
        if (
            self.environment_id != "StochasticTrack-v1"
            or self.max_episode_steps != 1800
        ):
            raise ValueError("Environment or episode horizon changed")
        if self.task != TaskSpecification(140.0, 0.0):
            raise ValueError("Canonical task changed")
        forbidden = (
            set(self.track_policy.historical_exploratory)
            | set(self.track_policy.validation)
            | set(self.track_policy.paper_final_test_reserved)
        )
        if set(self.screening.development_tracks) & forbidden:
            raise ValueError("Development search tracks overlap forbidden tracks")
        if self.allowed_candidate_imports != (
            "math",
            "benchmarks.llm_reward.reward_api",
        ):
            raise ValueError("Candidate import whitelist changed")

    @property
    def training_seeds(self) -> tuple[int, ...]:
        return (self.screening.training_seed,)

    @property
    def total_training_steps(self) -> int:
        return self.screening.simulator_transitions_per_candidate


def load_protocol(path: str | Path = DEFAULT_PROTOCOL_PATH) -> SearchProtocol:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Search protocol root must be an object")
    sac = dict(raw["sac"])
    sac["policy_network"] = tuple(sac["policy_network"])
    return SearchProtocol(
        schema_version=raw["schema_version"],
        protocol_id=raw["protocol_id"],
        phase=raw["phase"],
        environment_id=raw["environment_id"],
        max_episode_steps=raw["max_episode_steps"],
        task=TaskSpecification(**raw["task"]),
        search_budget=SearchBudget(
            **{
                **raw["search_budget"],
                "candidate_counts_by_generation": tuple(
                    raw["search_budget"]["candidate_counts_by_generation"]
                ),
            }
        ),
        screening=ScreeningProtocol(
            **{
                **raw["screening"],
                "evaluation_checkpoints": tuple(
                    raw["screening"]["evaluation_checkpoints"]
                ),
                "development_tracks": tuple(raw["screening"]["development_tracks"]),
            }
        ),
        sac=SACConfiguration(**sac),
        track_policy=TrackPolicy(
            **{name: tuple(values) for name, values in raw["track_policy"].items()}
        ),
        retry_policy=RetryPolicy(**raw["retry_policy"]),
        ranking=RankingProtocol(
            **{**raw["ranking"], "order": tuple(raw["ranking"]["order"])}
        ),
        final_evaluation=FinalEvaluationProtocol(
            **{
                **raw["final_evaluation"],
                "training_seeds": tuple(raw["final_evaluation"]["training_seeds"]),
                "validation_tracks": tuple(
                    raw["final_evaluation"]["validation_tracks"]
                ),
            }
        ),
        allowed_candidate_imports=tuple(raw["allowed_candidate_imports"]),
        information_barrier=tuple(raw["information_barrier"]),
    )


def protocol_sha256(protocol: SearchProtocol) -> str:
    payload = json.dumps(asdict(protocol), sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode()).hexdigest()
