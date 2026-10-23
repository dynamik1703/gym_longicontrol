"""Conservative, metadata-only feasibility analysis for historical tasks.

The module deliberately has no dependency on Gymnasium or an environment.  It
parses canonical task serializations that were already committed by completed
studies, verifies their fingerprints, and combines a mathematical lower bound
with an allow-list of immutable trajectory records.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal, localcontext
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]

PAPER_NUMERIC_SEEDS = frozenset(range(4000, 4018))
NUMERICAL_DISTANCE_MARGIN_M = Decimal("1e-9")
NUMERICAL_SPEED_MARGIN_M_S = Decimal("1e-8")

LEDGER_SOURCES = (
    "benchmarks/causal_mpc_v2/split_ledger.json",
    "benchmarks/constrained_v2_collector_replication/split_ledger.json",
    "benchmarks/constrained_v2_seed11_factorial_v3/split_ledger.json",
    "benchmarks/counterfactual_option_value_v1/split_ledger.json",
    "benchmarks/hybrid_rl_design/split_ledger.json",
    "benchmarks/hybrid_rl_validation_v2/split_ledger.json",
    "benchmarks/physical_feasibility_reference/split_ledger.json",
    "benchmarks/slacktrader_v1/split_ledger.json",
    "benchmarks/temporal_options_v1/split_ledger.json",
)

WITNESS_SOURCES = (
    "benchmarks/physical_feasibility_reference/development_results.json",
    "benchmarks/physical_feasibility_reference/holdout_results.json",
    "benchmarks/hybrid_rl_validation_v2/pair_ledger.json",
    "benchmarks/causal_mpc_v2/calibration_results.json",
    "benchmarks/causal_mpc_v2/capability_results.json",
    "benchmarks/slacktrader_v1/mpc_development_results.json",
    "benchmarks/temporal_options_v1/calibration_results.json",
    "benchmarks/temporal_options_v1/results.json",
    "benchmarks/counterfactual_option_value_v1/dataset_manifest.json",
)


class PaperSeedAccessError(ValueError):
    """Raised before any task payload associated with a Paper seed is read."""


class UnsupportedTaskSemantics(ValueError):
    """The proof is inapplicable; the task must remain unresolved."""


def assert_not_paper_seed(numeric_seed: int | None) -> None:
    if numeric_seed in PAPER_NUMERIC_SEEDS:
        raise PaperSeedAccessError(
            f"Paper seed {numeric_seed} is numeric-exclusion metadata only"
        )


def canonical_json_from_encoded(value: Any) -> str:
    """Serialize an already-canonical value without decoding float markers."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def decode_float_markers(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"__float_hex__"}:
            return float.fromhex(value["__float_hex__"])
        return {key: decode_float_markers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode_float_markers(item) for item in value]
    return value


def decimal_from_float(value: float) -> Decimal:
    return Decimal.from_float(float(value))


@dataclass(frozen=True)
class BoundResult:
    supported: bool
    lower_bound_s: Decimal | None
    continuous_limit_only_s: Decimal | None
    step_distance_upper_m: Decimal | None
    reason: str
    segment_terms_s: tuple[Decimal, ...] = ()


def conservative_discrete_lower_bound(task: dict[str, Any]) -> BoundResult:
    """Return a proof-valid lower bound for any zero-tolerance canonical route.

    A native step is a boundary step for a segment if it intersects the
    segment's entry or exit boundary. Monotone motion gives at most two such
    steps per segment. Their combined contribution inside the segment is at
    most twice a global step-distance bound. Every other step has both endpoint
    speeds bounded by the segment limit under right-endpoint checking, so its
    average speed is bounded by that limit (plus an explicit floating-point
    margin). Interior-step sets are disjoint across segments, hence their time
    lower bounds may be summed.
    """

    try:
        if task["schema"] != "longicontrol-content-fingerprint-v1":
            raise UnsupportedTaskSemantics("unsupported fingerprint schema")
        route = task["route"]
        semantics = task["task_semantics"]
        parameters = task["environment_parameters"]
        if route["infrastructure_observation"]["speed_limit_semantics"] != (
            "right-continuous-piecewise-constant"
        ):
            raise UnsupportedTaskSemantics("unsupported speed-limit semantics")
        if parameters["integration"]["name"] != (
            "constant-acceleration-explicit-step-v1"
        ):
            raise UnsupportedTaskSemantics("unsupported integration")
        if semantics["dynamics"]["name"] != "longicontrol-native-dynamics-v1":
            raise UnsupportedTaskSemantics("unsupported dynamics")
        initial = parameters["initial_vehicle_state"]
        if initial["position_m"] != 0.0 or initial["velocity_m_s"] != 0.0:
            raise UnsupportedTaskSemantics("unsupported initial state")
        tolerance = float(semantics["requirements"]["max_speed_violation_m_s"])
        if tolerance < 0.0:
            raise UnsupportedTaskSemantics("negative speed tolerance")
        dt = decimal_from_float(semantics["dynamics"]["time_step_s"])
        if dt <= 0:
            raise UnsupportedTaskSemantics("non-positive native timestep")
        low_v, high_v = parameters["vehicle"]["specifications"]["velocity_limits"]
        low_a, high_a = parameters["vehicle"]["specifications"]["acceleration_limits"]
        if low_v < 0.0 or high_v <= 0.0 or high_a < 0.0 or low_a > 0.0:
            raise UnsupportedTaskSemantics("unsupported vehicle bounds")
        vmax = decimal_from_float(high_v)
        amax = decimal_from_float(high_a)
        # This does not require the sharper average-endpoint cancellation.  It
        # directly bounds the exact source expression v*dt + 0.5*a*dt**2 and
        # adds 1 nm per step, far above the relevant binary64 roundoff.
        step_upper = (
            vmax * dt + Decimal("0.5") * amax * dt * dt + NUMERICAL_DISTANCE_MARGIN_M
        )
        continuous = Decimal(0)
        terms: list[Decimal] = []
        with localcontext() as context:
            context.prec = 50
            for segment in route["segments"]:
                length = decimal_from_float(segment["length_m"])
                limit = decimal_from_float(segment["speed_limit_m_s"])
                if length <= 0 or limit <= 0:
                    raise UnsupportedTaskSemantics("invalid segment")
                effective_limit = (
                    limit + decimal_from_float(tolerance) + NUMERICAL_SPEED_MARGIN_M_S
                )
                continuous += length / (limit + decimal_from_float(tolerance))
                interior_distance = max(Decimal(0), length - 2 * step_upper)
                terms.append(interior_distance / effective_limit)
            lower = sum(terms, Decimal(0))
        return BoundResult(
            supported=True,
            lower_bound_s=lower,
            continuous_limit_only_s=continuous,
            step_distance_upper_m=step_upper,
            reason="SUPPORTED_CONSERVATIVE_DISCRETE_BOUND",
            segment_terms_s=tuple(terms),
        )
    except (KeyError, TypeError, ValueError, UnsupportedTaskSemantics) as error:
        return BoundResult(False, None, None, None, str(error))


def classify_task(
    task: dict[str, Any] | None,
    *,
    witness_records: Iterable[dict[str, Any]] = (),
) -> tuple[str, BoundResult | None]:
    """Classify without converting failures or aggregate rates into evidence."""

    if task is None:
        return "UNRESOLVED", None
    bound = conservative_discrete_lower_bound(task)
    if bound.supported:
        deadline = decimal_from_float(
            task["task_semantics"]["requirements"]["max_time_s"]
        )
        if bound.lower_bound_s is not None and bound.lower_bound_s > deadline:
            if any(verified_witness(record, task) for record in witness_records):
                raise RuntimeError("trajectory witness contradicts infeasibility proof")
            return "PROVEN_INFEASIBLE", bound
    for record in witness_records:
        if verified_witness(record, task):
            return "FEASIBLE_WITNESS", bound
    return "UNRESOLVED", bound


def verified_witness(record: dict[str, Any], task: dict[str, Any]) -> bool:
    """Require exact original metrics; a failure or aggregate is never enough."""

    required = {
        "completed",
        "travel_time_s",
        "max_speed_violation_m_s",
        "route_sha256",
        "full_task_sha256",
        "source",
    }
    if not required <= record.keys() or record.get("aggregate_only", False):
        return False
    requirements = task["task_semantics"]["requirements"]
    return bool(
        record["completed"] is True
        and record["travel_time_s"] <= requirements["max_time_s"]
        and record["max_speed_violation_m_s"] <= requirements["max_speed_violation_m_s"]
        and record["route_sha256"] == task["route_sha256"]
        and record["full_task_sha256"] == task.get("_audit_full_task_sha256")
    )


def _walk_task_rows(value: Any, path: tuple[Any, ...] = ()):
    if isinstance(value, dict):
        if "canonical_task_serialization" in value and "full_task_sha256" in value:
            yield path, value
        for key, item in value.items():
            yield from _walk_task_rows(item, path + (key,))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_task_rows(item, path + (index,))


def load_historical_tasks(root: Path = REPOSITORY_ROOT) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    per_source_seen: set[tuple[str, str]] = set()
    for relative in LEDGER_SOURCES:
        source = root / relative
        payload = json.loads(source.read_text(encoding="utf-8"))
        for path, row in _walk_task_rows(payload):
            numeric_seed = row.get("numeric_seed")
            assert_not_paper_seed(numeric_seed)
            serialization = row["canonical_task_serialization"]
            full_hash = hashlib.sha256(serialization.encode("utf-8")).hexdigest()
            if full_hash != row["full_task_sha256"]:
                raise RuntimeError(f"full-task fingerprint mismatch in {relative}")
            encoded = json.loads(serialization)
            route_hash = hashlib.sha256(
                canonical_json_from_encoded(encoded["route"]).encode("utf-8")
            ).hexdigest()
            if route_hash != row["route_sha256"]:
                raise RuntimeError(f"route fingerprint mismatch in {relative}")
            task = decode_float_markers(encoded)
            # This audit-only identity is added only after the canonical bytes
            # above have independently reproduced both committed hashes.
            task["_audit_full_task_sha256"] = row["full_task_sha256"]
            key = row["full_task_sha256"]
            entry = tasks.setdefault(
                key,
                {
                    "full_task_sha256": key,
                    "route_sha256": row["route_sha256"],
                    "canonical_task": task,
                    "numeric_seeds": set(),
                    "occurrences": [],
                },
            )
            entry["numeric_seeds"].add(numeric_seed)
            occurrence_key = (relative, key)
            if occurrence_key in per_source_seen:
                continue
            per_source_seen.add(occurrence_key)
            path_text = "/".join(
                str(part) for part in path if not isinstance(part, int)
            )
            entry["occurrences"].append(
                {
                    "source": relative,
                    "panel_path": path_text,
                    "accepted": row.get("accepted") is True,
                    "rejection_reasons": row.get("rejection_reasons", []),
                    "numeric_seed": numeric_seed,
                }
            )
    return tasks


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk_dicts(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_dicts(item)


def load_verified_witnesses(
    tasks: dict[str, dict[str, Any]], root: Path = REPOSITORY_ROOT
) -> dict[str, list[dict[str, Any]]]:
    """Load only preregistered, task-linked original-environment outcomes."""

    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def add_seed_record(seed: int, metrics: dict[str, Any], source: str) -> None:
        assert_not_paper_seed(seed)
        completed = metrics.get("completed", metrics.get("completion"))
        travel = metrics.get("travel_time_s")
        violation = metrics.get("max_speed_violation_m_s")
        if completed is True and travel is not None and violation is not None:
            by_seed[seed].append(
                {
                    "completed": True,
                    "travel_time_s": travel,
                    "max_speed_violation_m_s": violation,
                    "source": source,
                }
            )

    # Strongest source: every outcome carries both task fingerprints.
    hybrid_source = WITNESS_SOURCES[2]
    hybrid = json.loads((root / hybrid_source).read_text(encoding="utf-8"))
    for pair in hybrid["pairs"]:
        outcome = pair.get("result", {}).get("outcome", {})
        if outcome:
            by_hash[outcome["full_task_sha256"]].append(
                {
                    "completed": outcome["completed"],
                    "travel_time_s": outcome["travel_time_s"],
                    "max_speed_violation_m_s": outcome["max_speed_violation_m_s"],
                    "route_sha256": outcome["route_sha256"],
                    "full_task_sha256": outcome["full_task_sha256"],
                    "source": hybrid_source,
                    "controller_id": outcome["controller_id"],
                }
            )

    generic_sources = (
        WITNESS_SOURCES[0],
        WITNESS_SOURCES[1],
        WITNESS_SOURCES[3],
        WITNESS_SOURCES[4],
        WITNESS_SOURCES[5],
        WITNESS_SOURCES[6],
        WITNESS_SOURCES[7],
    )
    for relative in generic_sources:
        payload = json.loads((root / relative).read_text(encoding="utf-8"))
        for item in _walk_dicts(payload):
            seed = item.get("numeric_seed", item.get("track_seed"))
            if not isinstance(seed, int):
                continue
            add_seed_record(seed, item, relative)

    # Counterfactual sampling retained one full original trajectory per route.
    manifest_source = WITNESS_SOURCES[8]
    manifest = json.loads((root / manifest_source).read_text(encoding="utf-8"))
    for row in manifest["route_results"]:
        metrics = dict(row["final_episode_metrics"])
        metrics["route_sha256"] = row["route_sha256"]
        add_seed_record(row["numeric_seed"], metrics, manifest_source)

    for full_hash, entry in tasks.items():
        route_hash = entry["route_sha256"]
        for seed in entry["numeric_seeds"]:
            for record in by_seed.get(seed, ()):
                if "route_sha256" in record and record["route_sha256"] != route_hash:
                    continue
                by_hash[full_hash].append(
                    {
                        **record,
                        "route_sha256": route_hash,
                        "full_task_sha256": full_hash,
                    }
                )
    return by_hash


def build_audit(root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    tasks = load_historical_tasks(root)
    witnesses = load_verified_witnesses(tasks, root)
    record_counts_by_source: dict[str, int] = defaultdict(int)
    verified_counts_by_source: dict[str, int] = defaultdict(int)
    for full_hash, records in witnesses.items():
        task = tasks[full_hash]["canonical_task"]
        for record in records:
            record_counts_by_source[record["source"]] += 1
            if verified_witness(record, task):
                verified_counts_by_source[record["source"]] += 1
    rows: list[dict[str, Any]] = []
    for full_hash, entry in sorted(tasks.items()):
        task = entry["canonical_task"]
        classification, bound = classify_task(
            task, witness_records=witnesses.get(full_hash, ())
        )
        requirements = task["task_semantics"]["requirements"]
        accepted = any(item["accepted"] for item in entry["occurrences"])
        segments = task["route"]["segments"]
        rows.append(
            {
                "full_task_sha256": full_hash,
                "route_sha256": entry["route_sha256"],
                "numeric_seeds": sorted(entry["numeric_seeds"]),
                "accepted_in_any_study": accepted,
                "study_occurrence_count": len(entry["occurrences"]),
                "occurrences": entry["occurrences"],
                "classification": classification,
                "deadline_s": requirements["max_time_s"],
                "speed_tolerance_m_s": requirements["max_speed_violation_m_s"],
                "continuous_speed_limit_only_s": (
                    float(bound.continuous_limit_only_s)
                    if bound and bound.continuous_limit_only_s is not None
                    else None
                ),
                "proven_discrete_lower_bound_s": (
                    float(bound.lower_bound_s)
                    if bound and bound.lower_bound_s is not None
                    else None
                ),
                "step_distance_upper_m": (
                    float(bound.step_distance_upper_m)
                    if bound and bound.step_distance_upper_m is not None
                    else None
                ),
                "proof_support": bound.reason if bound else "MISSING_TASK_CONTENT",
                "segments": [
                    {
                        "length_m": segment["length_m"],
                        "speed_limit_m_s": segment["speed_limit_m_s"],
                    }
                    for segment in segments
                ],
                "verified_witness_count": sum(
                    verified_witness(record, task)
                    for record in witnesses.get(full_hash, ())
                ),
                "witness_sources": sorted(
                    {
                        record["source"]
                        for record in witnesses.get(full_hash, ())
                        if verified_witness(record, task)
                    }
                ),
            }
        )

    counts: dict[str, int] = defaultdict(int)
    accepted_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["classification"]] += 1
        if row["accepted_in_any_study"]:
            accepted_counts[row["classification"]] += 1
    occurrence_count = sum(row["study_occurrence_count"] for row in rows)
    return {
        "schema_version": 1,
        "source_commit": "9802f9262ea7f753365fc4bdd20b1fe08379291d",
        "paper_state": "SEALED_UNINSPECTED",
        "paper_numeric_exclusion_only": list(range(4000, 4018)),
        "proof_method": "DISCRETE_SEGMENT_INTERIOR_BOUND_V1",
        "numerical_conservatism": {
            "per_step_distance_margin_m": float(NUMERICAL_DISTANCE_MARGIN_M),
            "interior_speed_margin_m_s": float(NUMERICAL_SPEED_MARGIN_M_S),
        },
        "coverage": {
            "source_ledgers": len(LEDGER_SOURCES),
            "unique_routes": len({row["route_sha256"] for row in rows}),
            "unique_full_tasks": len(rows),
            "study_task_occurrences": occurrence_count,
            "accepted_unique_tasks": sum(row["accepted_in_any_study"] for row in rows),
            "classification_all_inspectable": dict(sorted(counts.items())),
            "classification_accepted_tasks": dict(sorted(accepted_counts.items())),
            "trusted_historical_controller_task_records": sum(
                len(records) for records in witnesses.values()
            ),
            "verified_feasible_trajectory_records": sum(
                verified_counts_by_source.values()
            ),
            "controller_task_records_by_source": dict(
                sorted(record_counts_by_source.items())
            ),
            "verified_witness_records_by_source": dict(
                sorted(verified_counts_by_source.items())
            ),
        },
        "tasks": rows,
    }


CSV_FIELDS = (
    "full_task_sha256",
    "route_sha256",
    "numeric_seeds",
    "accepted_in_any_study",
    "study_occurrence_count",
    "classification",
    "deadline_s",
    "speed_tolerance_m_s",
    "continuous_speed_limit_only_s",
    "proven_discrete_lower_bound_s",
    "step_distance_upper_m",
    "verified_witness_count",
    "witness_sources",
    "source_ledgers",
)


def audit_csv(audit: dict[str, Any]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in audit["tasks"]:
        writer.writerow(
            {
                **{key: row.get(key) for key in CSV_FIELDS},
                "numeric_seeds": ";".join(map(str, row["numeric_seeds"])),
                "witness_sources": ";".join(row["witness_sources"]),
                "source_ledgers": ";".join(
                    sorted({item["source"] for item in row["occurrences"]})
                ),
            }
        )
    return stream.getvalue()


def generated_outputs(audit: dict[str, Any]) -> dict[Path, str]:
    return {
        PACKAGE_DIR / "feasibility_audit.json": json.dumps(
            audit, indent=2, sort_keys=True
        )
        + "\n",
        PACKAGE_DIR / "historical_task_status.csv": audit_csv(audit),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.write == arguments.check:
        parser.error("choose exactly one of --write or --check")
    outputs = generated_outputs(build_audit())
    if arguments.write:
        for path, content in outputs.items():
            path.write_text(content, encoding="utf-8")
        return 0
    stale = [
        str(path.relative_to(REPOSITORY_ROOT))
        for path, content in outputs.items()
        if not path.exists() or path.read_text(encoding="utf-8") != content
    ]
    if stale:
        raise SystemExit("stale generated audit artifacts: " + ", ".join(stale))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
