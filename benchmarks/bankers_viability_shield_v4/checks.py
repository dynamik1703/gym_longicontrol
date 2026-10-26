"""Charge quality and differential CPU; do not reopen scientific Attempts."""

import resource
import subprocess
import sys
import time

from .integrity import HERE, ROOT, source_identity, verify_sources
from .serialization import atomic_json, require


def main():
    require(not (HERE / "quality_checks.json").exists(), "quality already sealed")
    identity = source_identity()
    verify_sources(identity)
    start = time.process_time()
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    commands = [
        ["/private/tmp/bankers-v3-env/bin/ruff", "check", "--no-cache", str(HERE)],
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(HERE / "tests"),
            "--junitxml=" + str(HERE / "tests.xml"),
        ],
        [sys.executable, "-m", "benchmarks.bankers_viability_shield_v4.equivalence"],
    ]
    results = []
    for i, cmd in enumerate(commands):
        with (HERE / f"quality_{i}.log").open("xb") as log:
            status = subprocess.run(
                cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            ).returncode
        results.append(dict(command=cmd, exit_code=status))
        print("QUALITY", i, status, flush=True)
        if status:
            break
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (
        time.process_time()
        - start
        + after.ru_utime
        + after.ru_stime
        - before.ru_utime
        - before.ru_stime
    )
    verify_sources(identity)
    atomic_json(
        HERE / "quality_checks.json",
        dict(
            identity=identity,
            exit_code=max(r["exit_code"] for r in results),
            checks=results,
            cpu_s=cpu,
        ),
        immutable=True,
    )
    require(
        all(r["exit_code"] == 0 for r in results),
        "quality failed; inspect before any scientific preflight",
    )


if __name__ == "__main__":
    main()
