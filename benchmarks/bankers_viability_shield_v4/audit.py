"""Read-only inherited provenance audit, before opening any V4 experiment."""

import json
import platform
import subprocess
import sys

import gymnasium
import numpy as np

from .integrity import BASE, HERE, REMOTE, ROOT
from .serialization import atomic_json, digest, file_hash, require


def audit():
    require(not (HERE / "source_audit.json").exists(), "audit already sealed")
    require(
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        == BASE,
        "wrong initial base",
    )
    require(
        subprocess.check_output(
            ["git", "remote", "get-url", "--push", "gym_longicontrol_origin"],
            cwd=ROOT,
            text=True,
        ).strip()
        == REMOTE,
        "wrong remote",
    )
    entries = subprocess.check_output(
        ["git", "ls-tree", "-r", BASE], cwd=ROOT, text=True
    ).splitlines()
    names = [line.split("\t", 1)[1] for line in entries]
    # Opaque SHA-256 only; no Paper route is constructed or semantically opened.
    hashes = {
        n: file_hash(ROOT / n)
        for n in names
        if n.startswith("src/") or n.startswith("benchmarks/bankers_viability_shield_v")
    }
    value = dict(
        base_commit=BASE,
        source_tree=subprocess.check_output(
            ["git", "rev-parse", BASE + "^{tree}"], cwd=ROOT, text=True
        ).strip(),
        all_tracked_git_objects=entries,
        historical_actual_sha256=hashes,
        ordered_sha256=digest(hashes),
        worktrees=subprocess.check_output(
            ["git", "worktree", "list", "--porcelain"], cwd=ROOT, text=True
        ),
        remotes=subprocess.check_output(["git", "remote", "-v"], cwd=ROOT, text=True),
        runtime=dict(
            python=sys.version,
            executable=sys.executable,
            platform=platform.platform(),
            numpy=np.__version__,
            gymnasium=gymnasium.__version__,
        ),
        immutable_stops={
            "V1": "BANKERS_PRE_SHIELD_INTEGRITY_STOP",
            "V2": "BANKERS_V2_RESOURCE_STOP",
            "V3": "BANKERS_V3_RESOURCE_STOP",
        },
        v3_resource=json.loads(
            (
                HERE.parent / "bankers_viability_shield_v3/preparation_status.json"
            ).read_text()
        ),
        validation="UNTOUCHED",
        paper="SEALED_UNINSPECTED",
    )
    atomic_json(HERE / "source_audit.json", value, immutable=True)
    print("AUDIT", len(entries), len(hashes), value["source_tree"], flush=True)


if __name__ == "__main__":
    audit()
