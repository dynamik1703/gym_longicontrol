"""Authoritative two-worker exactly-once Diagnostic V2 transaction."""

from __future__ import annotations

import gzip
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .constants import (
    EXECUTION_MANIFEST,
    PAIR_LEDGER,
    PAIR_ROOT,
    RESULTS,
    ROOT,
)
from .integrity import full_pre_execution_gate
from .ledger import mark_completed, mark_started, validate_pair_ledger
from .utils import (
    atomic_bytes,
    atomic_json,
    canonical_bytes,
    canonical_sha256,
    file_sha256,
    load_json,
    utc_now,
)
from .verifier import verify_files, verify_native_evidence, verify_outcome

INTERPRETERS = {
    "C": Path(
        "/Users/romanliessner/gitlab/gym_longicontrol_feedback_chunk_rl/.venv/bin/python"
    ),
    "H": Path(
        "/Users/romanliessner/gitlab/gym_longicontrol_hybrid_rl_execution/.venv/bin/python"
    ),
}


class WorkerClient:
    def __init__(self, runtime_id: str, *, stderr_path: Path) -> None:
        self.runtime_id = runtime_id
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr_stream = stderr_path.open("ab")
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(ROOT)
        self.process = subprocess.Popen(
            (
                str(INTERPRETERS[runtime_id]),
                "-m",
                "benchmarks.feedback_chunk_diagnostic_v2.worker",
                "--runtime",
                runtime_id,
            ),
            cwd=ROOT,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr_stream,
            text=True,
            bufsize=1,
        )

    def ask(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.process.stdin is None or self.process.stdout is None:
            raise RuntimeError("worker pipes are unavailable")
        self.process.stdin.write(
            json.dumps(request, allow_nan=False, separators=(",", ":")) + "\n"
        )
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            code = self.process.poll()
            raise RuntimeError(f"worker {self.runtime_id} exited unexpectedly: {code}")
        response = json.loads(line)
        if response.get("command") in ("ERROR", "FATAL"):
            raise RuntimeError(
                f"worker {self.runtime_id} {response['error_type']}: "
                f"{response['error']}\n{response.get('traceback', '')}"
            )
        return response

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.ask({"command": "SHUTDOWN"})
            except BaseException:
                self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.stderr_stream.close()


def _evidence_bytes(rows: list[dict[str, Any]]) -> bytes:
    payload = b"".join(canonical_bytes(row) + b"\n" for row in rows)
    return gzip.compress(payload, compresslevel=6, mtime=0)


def _pair_paths(pair: dict[str, Any]) -> tuple[Path, Path]:
    ordinal = pair["identity"]["pair_ordinal"]
    stem = f"{ordinal:03d}-{pair['pair_id'][:16]}"
    return PAIR_ROOT / f"{stem}.json", PAIR_ROOT / f"{stem}.jsonl.gz"


def _persist_result(
    *,
    pair: dict[str, Any],
    outcome: dict[str, Any],
    rows: list[dict[str, Any]],
    execution_manifest_sha256: str,
) -> dict[str, Any]:
    reconstruction = verify_native_evidence(rows)
    verification = verify_outcome(outcome, reconstruction)
    result_path, evidence_path = _pair_paths(pair)
    evidence_payload = _evidence_bytes(rows)
    atomic_bytes(evidence_path, evidence_payload)
    result = {
        "schema_version": 1,
        "pair_id": pair["pair_id"],
        "identity": pair["identity"],
        "execution_manifest_sha256": execution_manifest_sha256,
        "outcome": outcome,
        "native_evidence_sha256": canonical_sha256(rows),
        "evidence_sha256": file_sha256(evidence_path),
        "evidence_path": str(evidence_path.relative_to(ROOT)),
        "independent_reconstruction": reconstruction,
        "independent_verification": verification,
    }
    result["result_sha256"] = canonical_sha256(result)
    atomic_json(result_path, result)
    checked = verify_files(result_path, evidence_path)
    return {
        "result_path": str(result_path.relative_to(ROOT)),
        "evidence_path": str(evidence_path.relative_to(ROOT)),
        "result_sha256": checked["result_sha256"],
        "evidence_sha256": checked["evidence_file_sha256"],
    }


def _record_integrity_stop(pair_id: str | None, error: BaseException) -> None:
    ledger = load_json(PAIR_LEDGER)
    ledger["state"] = "INTEGRITY_STOP"
    ledger["integrity_stop"] = {
        "at_utc": utc_now(),
        "pair_id": pair_id,
        "error_type": type(error).__name__,
        "error": str(error),
        "policy": "ambiguous STARTED pair is never rerun",
    }
    validate_pair_ledger(ledger)
    atomic_json(PAIR_LEDGER, ledger)
    atomic_json(
        ROOT / "runs/feedback-chunk-diagnostic-v2/incident.json",
        ledger["integrity_stop"],
    )


def seal_results(*, elapsed_seconds: float) -> dict[str, Any]:
    ledger = load_json(PAIR_LEDGER)
    counts = validate_pair_ledger(ledger)
    if counts.get("COMPLETED", 0) != ledger["pair_count"]:
        raise RuntimeError("cannot seal incomplete V2 results")
    outcomes = []
    retained_bytes = 0
    for pair in ledger["pairs"]:
        result_path = ROOT / pair["result_path"]
        evidence_path = ROOT / pair["evidence_path"]
        verify_files(result_path, evidence_path)
        result = load_json(result_path)
        outcomes.append({"identity": result["identity"], "outcome": result["outcome"]})
        retained_bytes += result_path.stat().st_size + evidence_path.stat().st_size
    result = {
        "schema_version": 1,
        "study": ledger["study"],
        "state": "COMPLETED_INDEPENDENTLY_VERIFIED",
        "pair_count": len(outcomes),
        "pair_schedule_sha256": ledger["schedule_sha256"],
        "active_evaluation_seconds": elapsed_seconds,
        "retained_pair_artifact_bytes": retained_bytes,
        "outcomes": outcomes,
    }
    result["results_sha256"] = canonical_sha256(result)
    atomic_json(RESULTS, result)
    return result


def execute() -> dict[str, Any]:
    manifest = load_json(EXECUTION_MANIFEST)
    manifest_sha256 = canonical_sha256(manifest)
    full_pre_execution_gate(manifest["scientific_source_bundle"])
    ledger = load_json(PAIR_LEDGER)
    counts = validate_pair_ledger(ledger)
    if counts.get("STARTED", 0):
        raise RuntimeError("integrity stop: an ambiguous STARTED pair exists")
    if counts.get("COMPLETED", 0):
        raise RuntimeError("transaction launch requires a wholly fresh ledger")
    workers: dict[str, WorkerClient] = {}
    active_started = time.perf_counter()
    current_pair_id: str | None = None
    try:
        for runtime_id in ("C", "H"):
            workers[runtime_id] = WorkerClient(
                runtime_id,
                stderr_path=ROOT
                / f"runs/feedback-chunk-diagnostic-v2/worker-{runtime_id}.stderr.log",
            )
            ready = workers[runtime_id].ask({"command": "READY"})
            expected = manifest["runtime_manifests"][runtime_id]["manifest_sha256"]
            if ready["runtime_manifest_sha256"] != expected:
                raise RuntimeError(f"worker {runtime_id} runtime changed")
        for ordinal in range(ledger["pair_count"]):
            live = load_json(PAIR_LEDGER)
            validate_pair_ledger(live)
            pair = live["pairs"][ordinal]
            if pair["state"] != "NOT_STARTED":
                raise RuntimeError("fresh schedule order changed during execution")
            current_pair_id = pair["pair_id"]
            worker = workers[pair["identity"]["worker_runtime"]]
            prepared = worker.ask(
                {
                    "command": "PREPARE",
                    "pair": pair,
                    "execution_manifest_sha256": manifest_sha256,
                }
            )
            if not prepared.get("checkpoint_loaded_and_verified"):
                raise RuntimeError("worker did not attest checkpoint readiness")
            token = prepared["ownership_token"]
            mark_started(
                PAIR_LEDGER,
                pair_id=pair["pair_id"],
                ownership_token=token,
                started_at_utc=utc_now(),
            )
            response = worker.ask(
                {
                    "command": "EXECUTE",
                    "pair_id": pair["pair_id"],
                    "ownership_token": token,
                }
            )
            artifacts = _persist_result(
                pair=pair,
                outcome=response["outcome"],
                rows=response["native_evidence"],
                execution_manifest_sha256=manifest_sha256,
            )
            mark_completed(
                PAIR_LEDGER,
                pair_id=pair["pair_id"],
                ownership_token=token,
                completed_at_utc=utc_now(),
                **artifacts,
            )
            current_pair_id = None
            if (ordinal + 1) % 12 == 0:
                print(
                    json.dumps(
                        {
                            "progress": ordinal + 1,
                            "total": ledger["pair_count"],
                            "elapsed_seconds": time.perf_counter() - active_started,
                        }
                    ),
                    flush=True,
                )
        elapsed = time.perf_counter() - active_started
        return seal_results(elapsed_seconds=elapsed)
    except BaseException as error:
        _record_integrity_stop(current_pair_id, error)
        raise
    finally:
        for worker in workers.values():
            worker.close()


if __name__ == "__main__":
    print(json.dumps(execute(), sort_keys=True))
