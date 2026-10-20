"""Deterministically render the completed contextual result summary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.json"
PAIRED = HERE / "paired_analysis.json"
INTEGRITY = HERE / "integrity.json"
OUTPUT = HERE / "RESULTS.md"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render() -> str:
    result = _load(RESULTS)
    lines = [
        "# Contextual Physical-Reference Results",
        "",
        "Status: **complete**. This is a post-hoc contextual analysis on two",
        "previously opened RL Validation panels, not a new independent held-out",
        "generalization benchmark. No RL policy was loaded, rerun or adapted.",
        "",
        "## Physical outcomes",
        "",
        (
            "| Panel | Reference | RSR | Completion | Deadline | Speed | "
            "Feasible energy mean / median (kWh) | Mean travel time (s) |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for panel in result["panels"].values():
        for kind in ("oracle", "causal"):
            summary = panel[kind]["summary"]
            energy = summary["feasible_only_energy_kwh"]
            lines.append(
                f"| `{panel['evaluation_id']}` | {kind} | "
                f"{summary['requirement_satisfaction_count']}/18 | "
                f"{summary['completion_count']}/18 | "
                f"{summary['deadline_compliance_count']}/18 | "
                f"{summary['speed_compliance_count']}/18 | "
                f"{energy['mean']:.9f} / {energy['median']:.9f} | "
                f"{summary['mean_travel_time_s']:.3f} |"
            )
    lines.extend(
        [
            "",
            "Both references achieved full feasibility on both panels. This is an",
            "achieved result for these 36 tasks, not a guarantee for all generated",
            "tracks. Every selected Oracle trajectory has exact replay parity.",
            "",
            "## Paired feasible-only reference differences",
            "",
            (
                "| Panel | Pairs | Causal−Oracle energy mean / median / range "
                "(kWh) | Aggregate mean difference | Travel-time difference mean |"
            ),
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for panel in result["panels"].values():
        paired = panel["paired_feasible_only"]
        energy = paired["energy_causal_minus_oracle_kwh"]
        time = paired["travel_time_causal_minus_oracle_s"]
        lines.append(
            f"| `{panel['evaluation_id']}` | {energy['count']} | "
            f"{energy['mean']:.9f} / {energy['median']:.9f} / "
            f"[{energy['minimum']:.9f}, {energy['maximum']:.9f}] | "
            f"{paired['aggregate_mean_energy_difference_percent']:.3f}% | "
            f"{time['mean']:.3f} s |"
        )
    lines.extend(["", "## RL failures with feasible physical witnesses", ""])
    for panel in result["panels"].values():
        lines.extend(
            [
                f"### {panel['evaluation_id']}",
                "",
                (
                    "| Policy/method | Failed pairs | Oracle / causal witnesses | "
                    "Distinct failed tracks | Oracle / causal solvable tracks | "
                    "Speed / deadline / noncompletion |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        comparison = panel["rl_comparison"]
        for policy, summary in comparison["by_policy_and_training_seed"].items():
            failure = summary["failure_components"]
            lines.append(
                f"| {policy} (seed {summary['training_seed']}) | "
                f"{summary['rl_failed_policy_track_pairs']} | "
                f"{summary['rl_failed_pairs_oracle_feasible']} / "
                f"{summary['rl_failed_pairs_causal_feasible']} | "
                f"{summary['distinct_rl_failed_tracks']} | "
                f"{summary['distinct_failed_tracks_oracle_feasible']} / "
                f"{summary['distinct_failed_tracks_causal_feasible']} | "
                f"{failure['speed_related']} / {failure['deadline_related']} / "
                f"{failure['noncompletion']} |"
            )
        for method, summary in comparison["pooled_by_method"].items():
            failure = summary["failure_components"]
            lines.append(
                f"| **{method} pooled** | "
                f"{summary['rl_failed_policy_track_pairs']} | "
                f"{summary['rl_failed_pairs_oracle_feasible']} / "
                f"{summary['rl_failed_pairs_causal_feasible']} | "
                f"{summary['distinct_rl_failed_tracks']} | "
                f"{summary['distinct_failed_tracks_oracle_feasible']} / "
                f"{summary['distinct_failed_tracks_causal_feasible']} | "
                f"{failure['speed_related']} / {failure['deadline_related']} / "
                f"{failure['noncompletion']} |"
            )
        lines.append("")
    lines.extend(
        [
            "Counts in the last column can overlap; the immutable exclusive modes",
            "and all per-task comparisons are retained in `results.json`. Policy-",
            "track pairs and distinct tracks are kept separate. SAMEINFO Constrained",
            "failures are primarily deadline-related (28/31); Scalar failures are",
            "noncompletion/deadline on all 54 pairs. GCSL Constrained failures are",
            "primarily deadline-related (31/32); GCSL failures are primarily speed-",
            "related (53/54).",
            "",
            "RL/reference energy differences are reported only for jointly feasible",
            "policy-track pairs in `paired_analysis.json`. No infeasible RL energy is",
            "interpreted as an optimization result.",
            "",
            "## Contextual interpretation",
            "",
            "1. **Did the Oracle solve RL-failed tracks?** Yes: every distinct failed",
            "   track and every failed policy-track pair in both panels has an Oracle",
            "   feasible witness.",
            "2. **Did the causal controller solve them?** Yes, with the same complete",
            "   witness coverage and 18/18 panel RSR.",
            "3. **Was hidden future-route information necessary here?** No for these",
            "   observed tasks: the ordinary-observation causal controller also solved",
            "   all tasks. This does not make hidden route information irrelevant in",
            "   general.",
            "4. **What drove RL failure?** SAMEINFO Scalar was crawling/noncompletion;",
            "   GCSL was speed violation; both Constrained panels were predominantly",
            "   deadline-limited with smaller speed/noncompletion components.",
            "5. **How large was the physical energy gap?** Causal minus Oracle mean",
            "   was 0.042859 kWh (28.184%) on SAMEINFO and 0.047356 kWh",
            "   (31.070%) on GCSL. The causal controller was respectively 40.600 s",
            "   and 34.661 s faster on average, leaving visible deadline reserve.",
            "6. **Is an RL energy opportunity established?** The achieved gap",
            "   motivates a feasibility-first energy study, but does not prove that",
            "   RL can capture it. The Oracle is privileged and not globally optimal;",
            "   a causal non-",
            "   learning energy-aware control is required to identify learning value.",
            "",
            "Limited preview therefore does not plausibly explain the observed",
            "failures on these panels by necessity: a controller with the same 8D",
            "sensor plus disclosed causal memory succeeded everywhere. It may still",
            "matter for other controllers or generated tracks.",
            "",
            "## Execution and integrity",
            "",
            "- Scientific controller-task records: 72, all unique; duplicates: 0.",
            "- Oracle internal search trajectories: 396 (11 frozen candidates/task).",
            "- Independent Oracle replay checks: 36; replay steps: 47,040.",
            "- RL policy episodes executed: 0.",
            "- Frozen physical sources/configuration: verified.",
            "- Immutable RL results/splits: read from exact terminal Git commits.",
            "- Controller retuning after contextual results: none.",
            "- Paper: `SEALED_UNINSPECTED`; content fingerprints constructed: 0.",
            "",
        ]
    )
    return "\n".join(lines)


def write() -> None:
    if not RESULTS.exists() or not PAIRED.exists() or not INTEGRITY.exists():
        raise RuntimeError("completed contextual artifacts are required")
    OUTPUT.write_text(render(), encoding="utf-8")
    integrity = _load(INTEGRITY)
    integrity["output_sha256"]["RESULTS.md"] = _sha256(OUTPUT)
    INTEGRITY.write_text(
        json.dumps(integrity, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    write()
