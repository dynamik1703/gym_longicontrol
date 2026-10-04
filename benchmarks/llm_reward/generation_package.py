"""Export exactly the information allowed in a fresh Reward Designer session."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .history import DEFAULT_HISTORY_PATH, candidate_by_id, load_history
from .protocol import DEFAULT_PROTOCOL_PATH, load_protocol

CONTEXT_FILES = ("TASK.md", "ALLOWED_SIGNALS.md", "REWARD_API.md")
FORBIDDEN_PACKAGE_TERMS = (
    "scalar v1",
    "scalar v2",
    "constrained v1",
    "constrained v2",
    "requirement-conditioned",
    "requirement_conditioned",
    "binary success reward",
    "binary_reward",
    "results.md",
    "results.json",
    "deadline_slack",
    "deadline-slack",
    "t_min",
)
PROTECTED_TRACK_IDS = tuple(range(3000, 3009)) + tuple(range(4000, 4018))


def assert_package_is_sanitized(text: str) -> None:
    lowered = text.lower()
    for term in FORBIDDEN_PACKAGE_TERMS:
        if term in lowered:
            raise ValueError(f"Forbidden research information in package: {term}")
    for track_id in PROTECTED_TRACK_IDS:
        if re.search(rf"(?<!\d){track_id}(?!\d)", text):
            raise ValueError(f"Protected track identifier in package: {track_id}")
    if "validation" in lowered or "paper-final" in lowered or "paper_final" in lowered:
        raise ValueError("Non-Development evaluation scope leaked into package")


def _context(context_directory: str | Path) -> str:
    root = Path(context_directory)
    parts = []
    for name in CONTEXT_FILES:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing sanitized context file: {path}")
        parts.extend((f"<!-- BEGIN {name} -->", path.read_text(encoding="utf-8")))
    return "\n\n".join(parts)


def build_generation_package(
    generation: int,
    *,
    protocol,
    context_directory: str | Path,
    history: dict | None = None,
) -> str:
    if generation < 0 or generation > protocol.search_budget.maximum_generation:
        raise ValueError("Generation lies outside the frozen search")
    candidate_count = protocol.search_budget.candidate_counts_by_generation[generation]
    sections = [
        "# Fresh-session Reward Designer package",
        "",
        _context(context_directory),
        "",
        "## Generation instructions",
        "",
        f"Generate exactly {candidate_count} new candidate reward files.",
        "Return each candidate as standalone Python source plus a short, visible "
        "design rationale. Do not provide hidden reasoning.",
        "Do not access any information outside this package.",
        "Every candidate will be screened with the same Stable-Baselines3 SAC "
        "learner for exactly 50,000 simulator transitions using one fixed seed.",
        "Only aggregate Development feedback may be returned in a later generation.",
        "Do not edit the task, API, algorithm, training budget or selection rule.",
    ]
    if generation == 0:
        sections.extend(
            (
                "At least three of the five rewards must differ in functional form, "
                "not only in numeric coefficients.",
                "No prior candidate or experiment feedback exists for this generation.",
            )
        )
    else:
        if history is None or history.get("status") != "OPEN":
            raise RuntimeError("Later-generation packages require an OPEN history")
        selection = next(
            (
                item
                for item in history["parent_selections"]
                if item["for_generation"] == generation
            ),
            None,
        )
        if selection is None:
            raise RuntimeError("Deterministic parents have not been selected")
        sections.extend(
            (
                "",
                "## Permitted parent material",
                "",
                "The following parents were selected mechanically. You may change "
                "coefficients or functional forms, add/remove components, or combine "
                "ideas. Do not infer unavailable experiment information.",
            )
        )
        for candidate_id in selection["candidate_ids"]:
            candidate = candidate_by_id(history, candidate_id)
            if candidate["reflection"] is None:
                raise RuntimeError(f"Parent lacks reflection: {candidate_id}")
            source = Path(candidate["source_path"]).read_text(encoding="utf-8")
            reflection = Path(candidate["reflection"]["json_path"]).read_text(
                encoding="utf-8"
            )
            sections.extend(
                (
                    "",
                    f"### Parent {candidate_id}",
                    "",
                    "Visible rationale:",
                    candidate["visible_rationale"],
                    "",
                    "```python",
                    source.rstrip(),
                    "```",
                    "",
                    "Sanitized Development reflection:",
                    "",
                    "```json",
                    json.dumps(json.loads(reflection), indent=2, sort_keys=True),
                    "```",
                )
            )
    package = "\n".join(sections).rstrip() + "\n"
    assert_package_is_sanitized(package)
    return package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY_PATH)
    parser.add_argument(
        "--context-dir", type=Path, default=Path(__file__).parent / "context"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    history = None
    if args.generation > 0:
        history = load_history(args.history, protocol)
    package = build_generation_package(
        args.generation,
        protocol=protocol,
        context_directory=args.context_dir,
        history=history,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(package, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
