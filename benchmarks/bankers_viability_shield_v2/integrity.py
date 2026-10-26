"""Immutable source identities, route protection, and exactly-once ledger."""

from __future__ import annotations

import json
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from benchmarks.bankers_viability_shield_v1.integrity import verify_sources as verify_v1
from benchmarks.content_split_audit.fingerprints import (
    canonical_json,
    fingerprint_track,
    full_task_payload,
)
from benchmarks.content_split_audit.generate_results import v2_semantics
from benchmarks.content_split_audit.streams import PAPER_SEEDS, track_from_seed

from .inherited_shield_adapter import METHODS
from .serialization import atomic_json, digest, file_hash, require

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = "0162477848bcfef2d1acea3e38a47b8fbd539299"
REMOTE = "https://github.com/dynamik1703/gym_longicontrol.git"
PROTOCOL_DOCS = (
    "RESEARCH_QUESTION.md",
    "PROOF_REVIEW.md",
    "BACKUP_INVARIANT_VERIFICATION.md",
    "DEVELOPMENT_PROTOCOL.md",
    "SERIALIZATION_CONTRACT.md",
)


def source_identity():
    files = sorted(
        list(HERE.glob("*.py"))
        + list((HERE / "tests").glob("*.py"))
        + [HERE / p for p in PROTOCOL_DOCS]
    )
    return {
        "base_commit": BASE,
        "contract": "bankers-evidence-v2.0",
        "implementation_sha256": digest(
            {str(p.relative_to(HERE)): file_hash(p) for p in files}
        ),
        "configuration_sha256": file_hash(HERE / "canonical.json"),
        "inherited_tree_sha256": file_hash(HERE / "source_audit.json"),
    }


def audit_sources():
    require(
        subprocess.check_output(
            ["git", "remote", "get-url", "--push", "gym_longicontrol_origin"],
            cwd=ROOT,
            text=True,
        ).strip()
        == REMOTE,
        "wrong push remote",
    )
    verify_v1()
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", BASE], cwd=ROOT, text=True
    ).splitlines()
    hashes = {p: file_hash(ROOT / p) for p in paths}
    require(
        subprocess.run(
            ["git", "diff", "--quiet", BASE, "--", *paths], cwd=ROOT
        ).returncode
        == 0,
        "tracked inherited source differs from frozen commit",
    )
    audit = {
        "base_commit": BASE,
        "verified_remote": REMOTE,
        "files": hashes,
        "historical_file_count": len(paths),
        "inherited_ordered_sha256": digest(hashes),
    }
    atomic_json(HERE / "source_audit.json", audit, immutable=True)
    return audit


def verify_sources(identity=None):
    audit = json.loads((HERE / "source_audit.json").read_text())
    for p, h in audit["files"].items():
        require(file_hash(ROOT / p) == h, "historical/source mutation " + p)
    if identity:
        require(source_identity() == identity, "post-freeze source/config mutation")
    return len(audit["files"])


def walk_rows(value):
    if isinstance(value, dict):
        # Protect rejected/reserved content too; it was already constructed.
        numeric = value.get("numeric_seed", value.get("reset_seed"))
        route, full = value.get("route_sha256"), value.get("full_task_sha256")
        if route is not None and full is not None:
            yield numeric, str(route), str(full)
        for child in value.values():
            yield from walk_rows(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_rows(child)


def protected():
    rows, sources = set(), []
    paths = set(ROOT.glob("benchmarks/**/*ledger*.json"))
    paths.add(ROOT / "benchmarks/content_split_audit/results.json")
    for path in sorted(paths):
        if HERE in path.parents:
            continue
        found = list(walk_rows(json.loads(path.read_text())))
        sources.append(
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": file_hash(path),
                "rows": len(found),
            }
        )
        rows.update(found)
    return rows, sources


def reserve_namespace():
    require(
        not (HERE / "namespace_reservation.json").exists(), "namespace already audited"
    )
    rows, sources = protected()
    numerics = {int(n) for n, _, _ in rows if n is not None}
    # All known scan reservations (even empty panels) are protected as blocks.
    reservations = []
    for path in sorted(ROOT.glob("benchmarks/**/*ledger*.json")):
        if HERE in path.parents:
            continue
        data = json.loads(path.read_text())

        def starts(v):
            if isinstance(v, dict):
                for k, x in v.items():
                    if k in (
                        "candidate_start",
                        "numeric_scan_start",
                        "scan_start",
                    ) and isinstance(x, int):
                        reservations.append(x)
                    starts(x)
            elif isinstance(v, list):
                for x in v:
                    starts(x)

        starts(data)
    block = 11_600_000
    skipped = []

    def occupied(b):
        return any(b <= n < b + 1_000_000 for n in numerics | set(reservations))

    # Requested start is a scan sub-block. Do not treat random canonical stream
    # seeds in its surrounding million as a namespace claim: only exact scan
    # range [start,start+10000) or explicit panel reservations occupy it.
    def requested_occupied():
        return any(block <= n < block + 10000 for n in numerics | set(reservations))

    if requested_occupied():
        skipped.append(block)
        block = 12_000_000
        while occupied(block):
            skipped.append(block)
            block += 1_000_000
    namespace = {
        "panel_id": "BANKERS_DEV_V2",
        "requested_start": 11_600_000,
        "selected_start": block,
        "skipped_occupied_blocks": skipped,
        "audited_before_task_construction": True,
        "protected_sources": sources,
        "protected_unique_rows": len(rows),
        "constructed_development_tasks": 0,
    }
    atomic_json(HERE / "namespace_reservation.json", namespace, immutable=True)
    return namespace


def build_split():
    require(not (HERE / "split_ledger.json").exists(), "split already exists")
    reservation = json.loads((HERE / "namespace_reservation.json").read_text())
    rows, sources = protected()
    numerics = {int(n) for n, _, _ in rows if n is not None} | set(PAPER_SEEDS)
    routes, fulls = {r for _, r, _ in rows}, {f for _, _, f in rows}
    selected, scanned = [], []
    seed = reservation["selected_start"]
    while len(selected) < 18:
        require(seed not in PAPER_SEEDS, "Paper sealed")
        track = track_from_seed(seed)
        fp = fingerprint_track(track, v2_semantics(), numeric_seed=seed)
        reasons = []
        if seed in numerics:
            reasons.append("NUMERIC_PROTECTED")
        if fp.route_sha256 in routes:
            reasons.append("ROUTE_PROTECTED")
        if fp.full_task_sha256 in fulls:
            reasons.append("FULL_TASK_PROTECTED")
        row = {
            "numeric_seed": seed,
            "route_sha256": fp.route_sha256,
            "full_task_sha256": fp.full_task_sha256,
            "accepted": not reasons,
            "rejection_reasons": reasons,
            "canonical_task_serialization": canonical_json(
                full_task_payload(track, v2_semantics())
            ),
        }
        scanned.append(row)
        if not reasons:
            selected.append(row)
            numerics.add(seed)
            routes.add(fp.route_sha256)
            fulls.add(fp.full_task_sha256)
        seed += 1
    result = {
        "panel_id": "BANKERS_DEV_V2",
        "ordered_tasks": selected,
        "ordered_sha256": digest(selected),
        "scanned_candidates": scanned,
        "protected_sources": sources,
        "protected_unique_rows": len(rows),
        "methods": list(METHODS),
        "performance_blind": True,
        "paper": "SEALED_UNINSPECTED",
        "state": "FROZEN_BEFORE_EXECUTION",
    }
    atomic_json(HERE / "split_ledger.json", result, immutable=True)
    return result


@contextmanager
def exclusive_writer(directory=HERE):
    path = Path(directory) / ".exclusive-writer"
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(fd, str(os.getpid()).encode())
        os.fsync(fd)
        yield
    finally:
        os.close(fd)
        path.unlink()


class PairLedger:
    def __init__(self, path, attempts, identity):
        self.path = Path(path)
        self.identity = identity
        expected = {
            "identity": identity,
            "schedule_sha256": digest(attempts),
            "pairs": {key: {"state": "NOT_STARTED"} for key in attempts},
        }
        if not self.path.exists():
            atomic_json(self.path, expected, immutable=True)
        self.data = json.loads(self.path.read_text())
        require(
            self.data["identity"] == identity
            and self.data["schedule_sha256"] == digest(attempts),
            "ledger source/config/schedule mismatch",
        )
        require(
            list(sorted(self.data["pairs"])) == list(sorted(attempts)),
            "ledger schedule membership",
        )
        require(
            not any(v["state"] == "STARTED" for v in self.data["pairs"].values()),
            "ambiguous STARTED pair; rerun forbidden",
        )

    def start(self, key):
        require(
            self.data["pairs"][key]["state"] == "NOT_STARTED", "pair rerun forbidden"
        )
        self.data["pairs"][key] = {
            "state": "STARTED",
            "started_unix": time.time(),
            "attempt_sha256": digest([key, self.identity]),
        }
        atomic_json(self.path, self.data)

    def complete(self, key, result):
        require(
            self.data["pairs"][key]["state"] == "STARTED", "completion without start"
        )
        self.data["pairs"][key].update(state="COMPLETED", result=result)
        atomic_json(self.path, self.data)
