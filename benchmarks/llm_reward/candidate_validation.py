"""Static checks, deterministic smoke validation and candidate ingestion."""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import math
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .history import (
    DEFAULT_HISTORY_PATH,
    candidate_by_id,
    load_history,
    next_candidate_id,
    save_history,
)
from .protocol import DEFAULT_PROTOCOL_PATH, load_protocol, protocol_sha256
from .reward_api import REWARD_CONTEXT_FIELDS, RewardContext, RewardOutput

SAFE_BUILTIN_CALLS = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "float",
    "int",
    "len",
    "max",
    "min",
    "pow",
    "round",
    "sum",
    "tuple",
}
SAFE_MATH_ATTRIBUTES = {
    "ceil",
    "copysign",
    "cos",
    "e",
    "erf",
    "exp",
    "expm1",
    "fabs",
    "floor",
    "fmod",
    "hypot",
    "isfinite",
    "log",
    "log1p",
    "pi",
    "sin",
    "sqrt",
    "tanh",
}
FORBIDDEN_IDENTIFIERS = {
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "open",
    "setattr",
    "track_id",
    "track_seed",
    "vars",
    "__builtins__",
    "__import__",
}
FORBIDDEN_TEXT = {
    "benchmarks.scalar",
    "benchmarks.constrained",
    "benchmarks.requirement_conditioned",
    "benchmarks.binary_reward",
    "results.json",
    "trajectories.json",
    "deadline_slack",
    "deadline-slack",
    "t_min",
    "optimal_action",
    "validation_seed",
}
FORBIDDEN_TRACK_IDS = (
    set(range(1000, 1009)) | set(range(3000, 3009)) | set(range(4000, 4018))
)


@dataclass(frozen=True)
class ValidationReport:
    valid: bool
    source_sha256: str
    errors: tuple[str, ...]
    source_line_count: int
    ast_node_count: int
    numeric_constant_count: int
    smoke_component_names: tuple[str, ...]


def source_sha256(source: str) -> str:
    return sha256(source.encode("utf-8")).hexdigest()


class CandidateSafetyVisitor(ast.NodeVisitor):
    def __init__(self):
        self.errors: list[str] = []
        self.functions: set[str] = set()

    def error(self, node: ast.AST, message: str) -> None:
        self.errors.append(f"line {getattr(node, 'lineno', 0)}: {message}")

    def visit_Import(self, node: ast.Import) -> None:
        if any(alias.name != "math" or alias.asname for alias in node.names):
            self.error(node, "only 'import math' is permitted")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level or node.module != "benchmarks.llm_reward.reward_api":
            self.error(node, "only the reward_api type import is permitted")
            return
        allowed = {"RewardContext", "RewardOutput"}
        if any(alias.name not in allowed or alias.asname for alias in node.names):
            self.error(node, "reward_api imports are limited to public API types")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.decorator_list:
            self.error(node, "function decorators are forbidden")
        arguments = node.args
        if (
            arguments.posonlyargs
            or arguments.kwonlyargs
            or arguments.vararg is not None
            or arguments.kwarg is not None
            or arguments.defaults
            or arguments.kw_defaults
        ):
            self.error(
                node, "function defaults, variadics and keyword-only args are forbidden"
            )
        if node.name == "compute_reward" and (
            len(arguments.args) != 1 or arguments.args[0].arg != "ctx"
        ):
            self.error(
                node, "compute_reward must accept exactly one argument named ctx"
            )
        if node.name != "compute_reward" and not node.name.startswith("_"):
            self.error(node, "helper functions must use a private name")
        self.functions.add(node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.error(node, "async functions are forbidden")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.error(node, "classes are forbidden")

    def visit_Global(self, node: ast.Global) -> None:
        self.error(node, "global mutation is forbidden")

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.error(node, "nonlocal mutation is forbidden")

    def visit_Delete(self, node: ast.Delete) -> None:
        self.error(node, "deletion is forbidden")

    def visit_For(self, node: ast.For) -> None:
        self.error(node, "loops are forbidden in per-step reward code")

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.error(node, "loops are forbidden in per-step reward code")

    def visit_While(self, node: ast.While) -> None:
        self.error(node, "loops are forbidden in per-step reward code")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("_"):
            self.error(node, "private or dunder attribute access is forbidden")
        if isinstance(node.value, ast.Name):
            if node.value.id == "ctx" and node.attr not in REWARD_CONTEXT_FIELDS:
                self.error(
                    node, f"RewardContext signal is not whitelisted: {node.attr}"
                )
            elif node.value.id == "math" and node.attr not in SAFE_MATH_ATTRIBUTES:
                self.error(node, f"math attribute is not whitelisted: {node.attr}")
            elif node.value.id not in {"ctx", "math"}:
                self.error(node, "attribute access is limited to ctx and math")
        else:
            self.error(node, "nested attribute access is forbidden")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_IDENTIFIERS or node.id.startswith("__"):
            self.error(node, f"forbidden identifier: {node.id}")

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, int) and node.value in FORBIDDEN_TRACK_IDS:
            self.error(node, "literal protected track identifier is forbidden")
        if isinstance(node.value, str):
            lowered = node.value.lower()
            for token in FORBIDDEN_TEXT:
                if token in lowered:
                    self.error(node, f"forbidden research reference: {token}")

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            permitted = SAFE_BUILTIN_CALLS | {"RewardOutput"} | self.functions
            if node.func.id not in permitted:
                self.error(node, f"call target is not whitelisted: {node.func.id}")
        elif isinstance(node.func, ast.Attribute):
            if not (
                isinstance(node.func.value, ast.Name)
                and node.func.value.id == "math"
                and node.func.attr in SAFE_MATH_ATTRIBUTES
            ):
                self.error(node, "only whitelisted math attribute calls are allowed")
        else:
            self.error(node, "dynamic call targets are forbidden")
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        if any(not isinstance(target, ast.Name) for target in node.targets):
            self.error(node, "assignment targets must be local names")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if not isinstance(node.target, ast.Name):
            self.error(node, "annotated assignment target must be a local name")
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if not isinstance(node.target, ast.Name):
            self.error(node, "augmented assignment target must be a local name")
        self.generic_visit(node)


def _static_errors(tree: ast.Module) -> tuple[str, ...]:
    errors = []
    allowed_top_level = (ast.Expr, ast.Import, ast.ImportFrom, ast.FunctionDef)
    for node in tree.body:
        if not isinstance(node, allowed_top_level):
            errors.append(
                f"line {getattr(node, 'lineno', 0)}: top-level state is forbidden"
            )
        if isinstance(node, ast.Expr) and not (
            isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)
        ):
            errors.append(
                f"line {getattr(node, 'lineno', 0)}: only a module docstring is allowed"
            )
    visitor = CandidateSafetyVisitor()
    visitor.functions.update(
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    )
    visitor.visit(tree)
    errors.extend(visitor.errors)
    functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    if functions.count("compute_reward") != 1:
        errors.append("exactly one top-level compute_reward function is required")
    return tuple(sorted(set(errors)))


def _safe_import(name, globals_=None, locals_=None, fromlist=(), level=0):
    del globals_, locals_
    if level:
        raise ImportError("relative imports are forbidden")
    if name == "math" and not fromlist:
        return math
    if name == "benchmarks.llm_reward.reward_api" and set(fromlist) <= {
        "RewardContext",
        "RewardOutput",
    }:
        import benchmarks.llm_reward.reward_api as reward_api

        return reward_api
    raise ImportError(f"candidate import is forbidden: {name}")


def _namespace(source: str) -> dict[str, Any]:
    safe_builtins = {name: getattr(builtins, name) for name in SAFE_BUILTIN_CALLS}
    safe_builtins["__import__"] = _safe_import
    namespace = {
        "__builtins__": MappingProxyType(safe_builtins),
        "__name__": "llm_reward_candidate",
        "RewardContext": RewardContext,
        "RewardOutput": RewardOutput,
    }
    exec(compile(source, "<llm-reward-candidate>", "exec"), namespace)  # noqa: S102
    return namespace


def smoke_contexts() -> tuple[RewardContext, ...]:
    base = {
        "position_m": 250.0,
        "previous_position_m": 249.0,
        "delta_position_m": 1.0,
        "velocity_m_s": 10.0,
        "previous_velocity_m_s": 9.9,
        "acceleration_m_s2": 1.0,
        "previous_acceleration_m_s2": 0.5,
        "action": 0.25,
        "speed_limit_m_s": 13.9,
        "future_speed_limit_1_m_s": 8.3,
        "future_speed_limit_2_m_s": 16.7,
        "distance_to_future_limit_1_m": 75.0,
        "distance_to_future_limit_2_m": 150.0,
        "step_energy_kwh": 0.001,
        "cumulative_energy_kwh": 0.2,
        "elapsed_time_s": 40.0,
        "dt_s": 0.1,
        "route_length_m": 1000.0,
        "time_budget_s": 140.0,
        "completed": False,
        "episode_ended": False,
    }
    return (
        RewardContext(**base),
        RewardContext(
            **{
                **base,
                "position_m": 1000.0,
                "previous_position_m": 999.5,
                "delta_position_m": 0.5,
                "velocity_m_s": 14.1,
                "speed_limit_m_s": 13.9,
                "step_energy_kwh": -0.0002,
                "elapsed_time_s": 139.9,
                "completed": True,
                "episode_ended": True,
            }
        ),
    )


def validate_candidate_source(source: str) -> ValidationReport:
    digest = source_sha256(source)
    line_count = len(source.splitlines())
    try:
        tree = ast.parse(source, filename="<llm-reward-candidate>")
    except SyntaxError as error:
        return ValidationReport(
            False,
            digest,
            (f"syntax error at line {error.lineno}: {error.msg}",),
            line_count,
            0,
            0,
            (),
        )
    nodes = tuple(ast.walk(tree))
    numeric_constant_count = sum(
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        for node in nodes
    )
    errors = list(_static_errors(tree))
    component_names: set[str] = set()
    if not errors:
        try:
            function = _namespace(source)["compute_reward"]
            for context in smoke_contexts():
                first = function(context)
                second = function(context)
                if not isinstance(first, RewardOutput):
                    raise TypeError("compute_reward must return RewardOutput")
                if not isinstance(second, RewardOutput):
                    raise TypeError("compute_reward must return RewardOutput")
                if first.reward != second.reward or dict(first.components) != dict(
                    second.components
                ):
                    raise ValueError("compute_reward is not deterministic")
                component_names.update(first.components)
        except Exception as error:  # noqa: BLE001 - archived validation diagnostic
            errors.append(f"runtime smoke failed: {type(error).__name__}: {error}")
    return ValidationReport(
        not errors,
        digest,
        tuple(sorted(set(errors))),
        line_count,
        len(nodes),
        numeric_constant_count,
        tuple(sorted(component_names)),
    )


def load_reward_function(path: str | Path):
    source = Path(path).read_text(encoding="utf-8")
    report = validate_candidate_source(source)
    if not report.valid:
        raise ValueError("Candidate failed validation: " + "; ".join(report.errors))
    return _namespace(source)["compute_reward"], report


def _candidate_metadata_path(root: Path, generation: int, candidate_id: str) -> Path:
    return root / f"g{generation}" / f"{candidate_id}.json"


def validate_visible_rationale(rationale: str) -> str:
    normalized = rationale.strip()
    if not normalized:
        raise ValueError("A short visible rationale is required")
    lowered = normalized.lower()
    forbidden = FORBIDDEN_TEXT | {
        "scalar v1",
        "scalar v2",
        "constrained v1",
        "constrained v2",
        "requirement-conditioned",
        "validation",
        "paper-final",
        "paper_final",
    }
    if any(term in lowered for term in forbidden):
        raise ValueError("Visible rationale contains forbidden research information")
    if any(
        re.search(rf"(?<!\d){track_id}(?!\d)", normalized)
        for track_id in FORBIDDEN_TRACK_IDS
    ):
        raise ValueError("Visible rationale contains a protected track identifier")
    return normalized


def repair_request_payload(
    candidate_id: str,
    report: ValidationReport,
    *,
    status: str = "repair_required",
) -> dict[str, Any]:
    if report.valid:
        raise ValueError("A valid candidate does not need a repair request")
    if status not in {"repair_required", "technical_failure"}:
        raise ValueError("Invalid failed-candidate status")
    return {
        "candidate_id": candidate_id,
        "status": status,
        "repair_feedback_scope": "validation-errors-only",
        "errors": list(report.errors),
    }


def ingest_candidate(
    source_path: str | Path,
    *,
    rationale: str,
    generation: int,
    history_path: str | Path,
    protocol,
    generations_root: str | Path,
    parent_candidate_ids: tuple[str, ...] = (),
    repair_candidate_id: str | None = None,
) -> tuple[str, ValidationReport]:
    history = load_history(history_path, protocol)
    if history["status"] != "OPEN":
        raise RuntimeError("Candidate ingestion requires an OPEN search")
    source = Path(source_path).read_text(encoding="utf-8")
    report = validate_candidate_source(source)
    rationale = validate_visible_rationale(rationale)
    updated = deepcopy(history)
    if repair_candidate_id is None:
        candidate_id = next_candidate_id(updated, protocol, generation)
        expected_parents = ()
        if generation:
            selection = next(
                item
                for item in updated["parent_selections"]
                if item["for_generation"] == generation
            )
            expected_parents = tuple(selection["candidate_ids"])
        if tuple(parent_candidate_ids) != expected_parents:
            raise ValueError("Candidate parents do not match deterministic selection")
        candidate = {
            "candidate_id": candidate_id,
            "generation": generation,
            "parent_candidate_ids": list(parent_candidate_ids),
            "visible_rationale": rationale,
            "source_sha256": report.source_sha256,
            "source_path": None,
            "metadata_path": None,
            "validation_attempts": [asdict(report)],
            "repair_attempt_count": 0,
            "status": "validated" if report.valid else "repair_required",
            "screening": None,
            "reflection": None,
            "ranking": None,
            "engineering_effort": {
                "reward_component_count": len(report.smoke_component_names),
                "numeric_constant_count": report.numeric_constant_count,
                "source_line_count": report.source_line_count,
                "ast_node_count": report.ast_node_count,
            },
        }
        updated["candidates"].append(candidate)
        updated["engineering_effort"]["generated_reward_candidate_count"] += 1
        updated["engineering_effort"]["generation_count_started"] = max(
            updated["engineering_effort"]["generation_count_started"], generation + 1
        )
    else:
        candidate_id = repair_candidate_id
        candidate = candidate_by_id(updated, candidate_id)
        if candidate["generation"] != generation:
            raise ValueError("Repair generation does not match candidate")
        if candidate["status"] != "repair_required":
            raise RuntimeError("Only a repair-required candidate can be repaired")
        if (
            candidate["repair_attempt_count"]
            >= protocol.retry_policy.maximum_repair_attempts_per_candidate
        ):
            raise RuntimeError("Candidate already consumed its single repair")
        if rationale != candidate["visible_rationale"]:
            raise ValueError("A repair must preserve the archived design rationale")
        candidate["repair_attempt_count"] += 1
        candidate["validation_attempts"].append(asdict(report))
        candidate["source_sha256"] = report.source_sha256
        candidate["status"] = "validated" if report.valid else "technical_failure"
        candidate["engineering_effort"] = {
            "reward_component_count": len(report.smoke_component_names),
            "numeric_constant_count": report.numeric_constant_count,
            "source_line_count": report.source_line_count,
            "ast_node_count": report.ast_node_count,
        }

    if report.valid:
        root = Path(generations_root)
        directory = root / f"g{generation}"
        directory.mkdir(parents=True, exist_ok=True)
        candidate_path = directory / f"{candidate_id}.py"
        metadata_path = _candidate_metadata_path(root, generation, candidate_id)
        candidate_path.write_text(source, encoding="utf-8")
        candidate["source_path"] = str(candidate_path)
        candidate["metadata_path"] = str(metadata_path)
        metadata = {
            "schema_version": 1,
            "protocol_id": protocol.protocol_id,
            "protocol_sha256": protocol_sha256(protocol),
            "candidate_id": candidate_id,
            "generation": generation,
            "parent_candidate_ids": list(candidate["parent_candidate_ids"]),
            "visible_rationale": candidate["visible_rationale"],
            "source_sha256": report.source_sha256,
            "validation": asdict(report),
            "repair_attempt_count": candidate["repair_attempt_count"],
            "status": "validated",
            "engineering_effort": candidate["engineering_effort"],
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    save_history(history_path, updated, protocol)
    return candidate_id, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--rationale-file", type=Path, required=True)
    parser.add_argument("--parent", action="append", default=[])
    parser.add_argument("--repair-candidate-id")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument(
        "--generations-root", type=Path, default=Path(__file__).with_name("generations")
    )
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    candidate_id, report = ingest_candidate(
        args.source,
        rationale=args.rationale_file.read_text(encoding="utf-8"),
        generation=args.generation,
        history_path=args.history,
        protocol=protocol,
        generations_root=args.generations_root,
        parent_candidate_ids=tuple(args.parent),
        repair_candidate_id=args.repair_candidate_id,
    )
    if report.valid:
        payload = {"candidate_id": candidate_id, **asdict(report)}
    else:
        status = candidate_by_id(load_history(args.history, protocol), candidate_id)[
            "status"
        ]
        payload = repair_request_payload(candidate_id, report, status=status)
    print(json.dumps(payload, sort_keys=True))
    return 0 if report.valid else 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
