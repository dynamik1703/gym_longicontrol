"""Frozen-source checks, durable exclusive ledger and Policy-R selection."""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from benchmarks.content_split_audit.fingerprints import (
    canonical_json,
    fingerprint_track,
    full_task_payload,
)
from benchmarks.content_split_audit.generate_results import v2_semantics
from benchmarks.content_split_audit.streams import PAPER_SEEDS, track_from_seed

from .backup_policy import SafetyIntegrityError
from .certificate import HERE, digest, identities

ROOT = HERE.parents[1]
START = 11_500_000
PANEL = "BANKERS_DEV_V1"


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, allow_nan=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_sources():
    audit = json.loads((HERE / "source_audit.json").read_text())
    for relative, expected in audit["hashes"].items():
        if file_sha256(ROOT / relative) != expected:
            raise SafetyIntegrityError("inherited source mutation: " + relative)
    return {"checked_files": len(audit["hashes"]), "identity": identities()}


def walk_rows(value):
    if isinstance(value, dict):
        if value.get("accepted", True):
            numeric = value.get("numeric_seed", value.get("reset_seed"))
            route, full = value.get("route_sha256"), value.get("full_task_sha256")
            if route is not None and full is not None:
                yield numeric, str(route), str(full)
        for child in value.values():
            yield from walk_rows(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_rows(child)


def protected_rows():
    rows, sources = set(), []
    # Ledgers include complete canonical stream content from prior split studies.
    # Also inspect existing static content audit results, without constructing
    # any new historical or sealed task.
    paths = set(ROOT.glob("benchmarks/**/*ledger*.json"))
    paths.add(ROOT / "benchmarks/content_split_audit/results.json")
    for path in sorted(paths):
        if HERE in path.parents:
            continue
        found = list(walk_rows(json.loads(path.read_text())))
        if found:
            sources.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": file_sha256(path),
                    "rows": len(found),
                }
            )
            rows.update(found)
    return rows, sources


def build_split():
    if (HERE / "split_ledger.json").exists():
        raise SafetyIntegrityError("split already materialized")
    rows, sources = protected_rows()
    numerics = {int(n) for n, _, _ in rows if n is not None} | set(PAPER_SEEDS)
    routes = {r for _, r, _ in rows}
    fulls = {f for _, _, f in rows}
    namespace_hits = []
    for path in sorted(ROOT.glob("benchmarks/**/*.json")):
        if HERE in path.parents:
            continue
        if PANEL in path.read_text():
            namespace_hits.append(str(path.relative_to(ROOT)))
    if namespace_hits or any(START <= n < START + 18 for n in numerics):
        raise SafetyIntegrityError("namespace or numeric scan already occupied")
    selected, scanned = [], []
    seed = START
    while len(selected) < 18:
        if seed in PAPER_SEEDS:
            raise SafetyIntegrityError("PAPER_SEALED_UNINSPECTED")
        track = track_from_seed(seed)
        fp = fingerprint_track(track, v2_semantics(), numeric_seed=seed)
        reasons = []
        if seed in numerics:
            reasons.append("NUMERIC_RESERVED")
        if fp.route_sha256 in routes:
            reasons.append("ROUTE_RESERVED")
        if fp.full_task_sha256 in fulls:
            reasons.append("FULL_TASK_RESERVED")
        row = {
            "numeric_seed": seed,
            "route_sha256": fp.route_sha256,
            "full_task_sha256": fp.full_task_sha256,
            "canonical_task_serialization": canonical_json(
                full_task_payload(track, v2_semantics())
            ),
            "accepted": not reasons,
            "rejection_reasons": reasons,
        }
        scanned.append(row)
        if not reasons:
            selected.append(row)
            numerics.add(seed)
            routes.add(fp.route_sha256)
            fulls.add(fp.full_task_sha256)
        seed += 1
    ledger = {
        "panel_id": PANEL,
        "state": "FROZEN_BEFORE_EVALUATION",
        "ordered_tasks": selected,
        "ordered_sha256": digest(selected),
        "scanned_candidates": scanned,
        "protected_unique_rows": len(rows),
        "protected_sources": sources,
        "selection": "first 18 ascending numeric, route/full-task clean",
        "performance_blind": True,
        "namespace_hits": namespace_hits,
        "paper": {
            "state": "SEALED_UNINSPECTED",
            "numeric_exclusion_only": list(PAPER_SEEDS),
            "constructed": 0,
            "episodes": 0,
        },
        "validation_episodes": 0,
    }
    checks = {
        "exactly_18": len(selected) == 18,
        "unique_numeric": len({r["numeric_seed"] for r in selected}) == 18,
        "unique_routes": len({r["route_sha256"] for r in selected}) == 18,
        "unique_full_tasks": len({r["full_task_sha256"] for r in selected}) == 18,
        "historical_numeric_disjoint": not (
            {r["numeric_seed"] for r in selected}
            & {int(n) for n, _, _ in rows if n is not None}
        ),
        "historical_route_disjoint": not (
            {r["route_sha256"] for r in selected} & {r for _, r, _ in rows}
        ),
        "historical_full_task_disjoint": not (
            {r["full_task_sha256"] for r in selected} & {f for _, _, f in rows}
        ),
        "paper_numeric_disjoint": not (
            {r["numeric_seed"] for r in selected} & set(PAPER_SEEDS)
        ),
    }
    if not all(checks.values()):
        raise SafetyIntegrityError("Policy-R split verification failed")
    ledger["checks"] = checks
    atomic_json(HERE / "split_ledger.json", ledger)
    return ledger


def verify_task(row):
    seed = int(row["numeric_seed"])
    if seed in PAPER_SEEDS:
        raise SafetyIntegrityError("PAPER_SEALED_UNINSPECTED")
    fp = fingerprint_track(track_from_seed(seed), v2_semantics(), numeric_seed=seed)
    if (
        fp.route_sha256 != row["route_sha256"]
        or fp.full_task_sha256 != row["full_task_sha256"]
    ):
        raise SafetyIntegrityError("runtime task identity mismatch")


@contextmanager
def exclusive_writer(directory):
    path = Path(directory) / ".exclusive-writer"
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, str(os.getpid()).encode())
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        path.unlink()


class PairLedger:
    def __init__(self, path, tasks, manifest):
        self.path = Path(path)
        self.manifest = manifest
        schedule = [
            (method, task["numeric_seed"])
            for task in tasks
            for method in ("CAUSAL60", "OLD_POST", "BANKERS_PRE")
        ]
        self.schedule_hash = digest(schedule)
        if not self.path.exists():
            atomic_json(
                self.path,
                {
                    "manifest": manifest,
                    "schedule_sha256": self.schedule_hash,
                    "pairs": {
                        f"{m}__{s}": {"state": "NOT_STARTED"} for m, s in schedule
                    },
                },
            )
        self.data = json.loads(self.path.read_text())
        if (
            self.data["manifest"] != manifest
            or self.data["schedule_sha256"] != self.schedule_hash
        ):
            raise SafetyIntegrityError("resume source/config/schedule mismatch")
        if any(v["state"] == "STARTED" for v in self.data["pairs"].values()):
            raise SafetyIntegrityError("ambiguous STARTED pair; rerun forbidden")

    def start(self, key):
        if self.data["pairs"][key]["state"] != "NOT_STARTED":
            raise SafetyIntegrityError("pair already started or completed")
        self.data["pairs"][key] = {
            "state": "STARTED",
            "started_unix": time.time(),
            "identity": digest([self.manifest, key]),
        }
        atomic_json(self.path, self.data)

    def complete(self, key, result):
        if self.data["pairs"][key]["state"] != "STARTED":
            raise SafetyIntegrityError("completion without STARTED")
        self.data["pairs"][key].update(state="COMPLETED", result=result)
        atomic_json(self.path, self.data)
