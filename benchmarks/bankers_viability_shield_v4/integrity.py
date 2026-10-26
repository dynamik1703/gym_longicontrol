"""Isolated V4 provenance, exclusive ownership and immutable STARTED records."""

import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import gymnasium
import numpy as np

from benchmarks.bankers_viability_shield_v2.integrity import PairLedger as OldPairLedger

from .serialization import atomic_json, digest, file_hash, require

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = "b60549e95e31601039329ca8de041977da0155fe"
REMOTE = "https://github.com/dynamik1703/gym_longicontrol.git"
PROTOCOLS = (
    "COMPACT_CERTIFICATE_SCHEMA.md",
    "RECONSTRUCTION_PROOF.md",
    "RESOURCE_PROTOCOL.md",
    "DEVELOPMENT_PROTOCOL.md",
)


def runtime_current():
    return dict(
        python=sys.version,
        executable=sys.executable,
        numpy=np.__version__,
        gymnasium=gymnasium.__version__,
        machine=platform.machine(),
    )


def source_identity():
    paths = sorted(
        [
            *HERE.glob("*.py"),
            *(HERE / "tests").glob("*.py"),
            *[HERE / p for p in PROTOCOLS],
        ]
    )
    return dict(
        runtime_sha256=file_hash(HERE / "runtime_manifest.json"),
        base_commit=BASE,
        contract="bankers-evidence-v4.0",
        implementation_sha256=digest(
            {str(p.relative_to(HERE)): file_hash(p) for p in paths}
        ),
        configuration_sha256=file_hash(HERE / "canonical.json"),
        inherited_tree_sha256=file_hash(HERE / "source_audit.json"),
    )


def verify_sources(identity=None):
    remote = subprocess.check_output(
        ["git", "remote", "get-url", "--push", "gym_longicontrol_origin"],
        cwd=ROOT,
        text=True,
    ).strip()
    require(remote == REMOTE, "wrong push remote")
    require(
        subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                BASE,
                "--",
                ".",
                ":(exclude)benchmarks/bankers_viability_shield_v4",
            ],
            cwd=ROOT,
        ).returncode
        == 0,
        "historical source/artifact mutation",
    )
    audit = json.loads((HERE / "source_audit.json").read_text())
    for name, expected in audit["historical_actual_sha256"].items():
        require(file_hash(ROOT / name) == expected, "historical mutation " + name)
    require(
        json.loads((HERE / "runtime_manifest.json").read_text()) == runtime_current(),
        "scientific runtime mutation",
    )
    if identity:
        require(source_identity() == identity, "post-freeze source/config mutation")
    return dict(
        historical_tracked_files=len(audit["all_tracked_git_objects"]),
        historical_hashed_files=len(audit["historical_actual_sha256"]),
        source_tree=audit["source_tree"],
    )


def require_committed():
    paths = [
        *HERE.glob("*.py"),
        *(HERE / "tests").glob("*.py"),
        *[HERE / p for p in PROTOCOLS],
        HERE / "canonical.json",
        HERE / "runtime_manifest.json",
    ]
    for p in paths:
        stored = subprocess.check_output(
            ["git", "show", "HEAD:" + str(p.relative_to(ROOT))], cwd=ROOT
        )
        require(stored == p.read_bytes(), "source/protocol not committed " + p.name)
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


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


class PairLedger(OldPairLedger):
    def __init__(self, path, attempts, identity):
        super().__init__(path, attempts, identity)
        for key, value in self.data["pairs"].items():
            marker = self.path.parent / "started" / f"{key}.json"
            if marker.exists():
                require(
                    value["state"] == "COMPLETED",
                    "ambiguous immutable STARTED marker; rerun forbidden",
                )
                require(
                    file_hash(marker) == value["started_file_sha256"],
                    "STARTED timestamp/identity mutation",
                )
            if value["state"] == "COMPLETED":
                require(marker.exists(), "missing immutable STARTED")
                result = self.path.parent / "sealed_results" / f"{key}.json"
                require(
                    file_hash(result) == value["result_file_sha256"],
                    "sealed result mutation",
                )
                require(
                    json.loads(result.read_text()) == value["result"],
                    "result/ledger mismatch",
                )

    def start(self, key):
        require(
            self.data["pairs"][key]["state"] == "NOT_STARTED", "pair rerun forbidden"
        )
        atomic_json(
            self.path.parent / "started" / f"{key}.json",
            dict(
                attempt=key,
                identity=self.identity,
                started_unix=time.time(),
                recovery="NONE_RERUN_FORBIDDEN",
            ),
            immutable=True,
        )
        super().start(key)
        self.data["pairs"][key]["started_file_sha256"] = file_hash(
            self.path.parent / "started" / f"{key}.json"
        )
        atomic_json(self.path, self.data)

    def complete(self, key, result):
        require(
            self.data["pairs"][key]["state"] == "STARTED", "completion without start"
        )
        path = self.path.parent / "sealed_results" / f"{key}.json"
        atomic_json(path, result, immutable=True)
        self.data["pairs"][key]["result_file_sha256"] = file_hash(path)
        super().complete(key, result)


def guarded_track(seed):
    require(type(seed) is int and not 4000 <= seed <= 4017, "PAPER_SEALED_UNINSPECTED")
    from benchmarks.content_split_audit.streams import track_from_seed

    return track_from_seed(seed)


def build_split():
    """Read only already available fingerprint metadata; no historical resets."""
    from benchmarks.bankers_viability_shield_v2.integrity import protected, walk_rows
    from benchmarks.content_split_audit.fingerprints import (
        canonical_json,
        fingerprint_track,
        full_task_payload,
    )
    from benchmarks.content_split_audit.generate_results import v2_semantics

    prep = json.loads((HERE / "preparation_status.json").read_text())
    require(prep["all_gates_passed"], "Development gate closed")
    verify_sources(prep["identity"])
    require(not (HERE / "split_ledger.json").exists(), "split already constructed")
    rows, sources = protected()
    # protected() is inherited; it excludes V2 itself. Explicitly include V2
    # namespace/ledgers and all reservation metadata, including unopened panels.
    reservations = []
    protected_numbers = set()
    extra_routes = set()
    extra_fulls = set()
    paths = set(ROOT.glob("benchmarks/**/*ledger*.json")) | set(
        ROOT.glob("benchmarks/**/*reservation*.json")
    )

    def collect(v):
        if isinstance(v, dict):
            for key in ("numeric_seed", "reset_seed"):
                if type(v.get(key)) is int:
                    protected_numbers.add(v[key])
            if isinstance(v.get("route_sha256"), str):
                extra_routes.add(v["route_sha256"])
            if isinstance(v.get("full_task_sha256"), str):
                extra_fulls.add(v["full_task_sha256"])
            for k, x in v.items():
                if k in (
                    "candidate_start",
                    "numeric_scan_start",
                    "scan_start",
                    "selected_start",
                    "requested_start",
                ) and isinstance(x, int):
                    reservations.append(x)
                collect(x)
        elif isinstance(v, list):
            for x in v:
                collect(x)

    for p in sorted(paths):
        if HERE in p.parents:
            continue
        value = json.loads(p.read_text())
        if not any(x["path"] == str(p.relative_to(ROOT)) for x in sources):
            sources.append(
                dict(
                    path=str(p.relative_to(ROOT)),
                    sha256=file_hash(p),
                    rows=len(list(walk_rows(value))),
                )
            )
        rows.update(walk_rows(value))
        collect(value)
    numerics = {int(n) for n, _, _ in rows if n is not None} | set(range(4000, 4018))
    numerics.update(protected_numbers)
    routes = {r for _, r, _ in rows} | extra_routes
    fulls = {f for _, _, f in rows} | extra_fulls
    start = 11800000
    skipped = []

    # Exact requested scan reservation; canonical random seeds do not reserve a million.
    def occupied(lo, hi):
        return any(lo <= n < hi for n in numerics | set(reservations))

    if occupied(start, start + 10000):
        skipped.append(start)
        start = 12000000
        while occupied(start, start + 1000000):
            skipped.append(start)
            start += 1000000
    namespace = dict(
        panel_id="BANKERS_DEV_V4",
        requested_start=11800000,
        selected_start=start,
        collision_blocks=skipped,
        protected_metadata_sources=sources,
        protected_rows=len(rows),
    )
    atomic_json(HERE / "namespace_reservation.json", namespace, immutable=True)
    accepted = []
    scanned = []
    seed = start
    while len(accepted) < 18:
        # Policy-R numeric exclusion precedes construction, even outside Paper.
        if seed in numerics:
            scanned.append(
                dict(
                    numeric_seed=seed,
                    accepted=False,
                    rejection_reasons=["NUMERIC_PROTECTED"],
                )
            )
            seed += 1
            continue
        track = guarded_track(seed)
        fp = fingerprint_track(track, v2_semantics(), numeric_seed=seed)
        reasons = []
        if fp.route_sha256 in routes:
            reasons.append("ROUTE_PROTECTED")
        if fp.full_task_sha256 in fulls:
            reasons.append("FULL_TASK_PROTECTED")
        row = dict(
            numeric_seed=seed,
            route_sha256=fp.route_sha256,
            full_task_sha256=fp.full_task_sha256,
            accepted=not reasons,
            rejection_reasons=reasons,
            canonical_task_serialization=canonical_json(
                full_task_payload(track, v2_semantics())
            ),
        )
        scanned.append(row)
        if not reasons:
            accepted.append(row)
            numerics.add(seed)
            routes.add(fp.route_sha256)
            fulls.add(fp.full_task_sha256)
        seed += 1
    split = dict(
        panel_id="BANKERS_DEV_V4",
        ordered_tasks=accepted,
        ordered_sha256=digest(accepted),
        scanned_candidates=scanned,
        policy="Policy-R",
        selection="FIRST_18_NO_PERFORMANCE_FILTER",
        paper="SEALED_UNINSPECTED",
        state="FROZEN_BEFORE_EXECUTION",
        protected_metadata_sources=sources,
    )
    atomic_json(HERE / "split_ledger.json", split, immutable=True)
    return split
