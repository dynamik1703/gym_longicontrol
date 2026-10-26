"""V3 evidence envelope; compact transcripts and complete native production state."""

import gzip
import hashlib
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from benchmarks.bankers_viability_shield_v2.serialization import (
    EvidenceError,
    atomic_json,
    canonical_bytes,
    digest,
    file_hash,
    fsync_directory,
    normalize,
    require,
    strict_load,
)

CONTRACT = "bankers-evidence-v3.0"
__all__ = [
    "CONTRACT",
    "EvidenceError",
    "atomic_json",
    "canonical_bytes",
    "digest",
    "file_hash",
    "normalize",
    "require",
    "strict_load",
    "write_evidence",
    "read_evidence",
    "native_journal",
]
REQUIRED = set(
    (
        "step method causal_pre_state observation memory "
        "causal60_proposal proposal_action "
        "certificates safe_action_mask selected_index "
        "selected_certificate causal60_proposal_certificate "
        "executed_action predicted_next_state "
        "actual_next_state actual_power_kw "
        "signed_energy_increment_kwh actual_speed_limit_m_s "
        "speed_excess_m_s "
        "terminated truncated legacy_proposal_check "
        "diagnostic_sampled inherited_comparison authority "
        "timing"
    ).split()
)


def validate_records(rows):
    require(bool(rows), "empty evidence")
    for i, r in enumerate(rows):
        require(set(r) == REQUIRED, "missing or unknown safety fields")
        require(type(r["step"]) is int and r["step"] == i, "step sequence")
        require(len(r["observation"]) == 8, "ordinary dimension")
        require(
            r["method"]
            in (
                "CAUSAL60",
                "LEGACY_NATIVE_POST_GREEDY",
                "BANKERS_GREEDY",
                "BANKERS_MASKED_CAUSAL60",
                "V3_SYNTHETIC_MINIMAL",
            ),
            "method",
        )
        certs = r["certificates"]
        require(len(certs) in (0, 21), "certificate scope")
        require(r["safe_action_mask"] == [c["accepted"] for c in certs], "mask meaning")
        require(all(type(v) is bool for v in r["safe_action_mask"]), "mask booleans")
        for k in ("causal_pre_state", "actual_next_state", "predicted_next_state"):
            require(
                set(r[k])
                == {
                    "position_m",
                    "velocity_m_s",
                    "acceleration_m_s2",
                    "jerk_m_s3",
                    "elapsed_time_s",
                    "total_energy_kwh",
                },
                "full native state",
            )
        require(
            type(r["terminated"]) is bool and type(r["truncated"]) is bool,
            "lifecycle types",
        )
    normalize(rows)


def write_evidence(path, records, identity, attempt):
    rows = normalize(records)
    validate_records(rows)
    header = dict(
        contract=CONTRACT, identity=identity, attempt=attempt, record_count=len(rows)
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    h = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
                for value in (header, *rows):
                    line = canonical_bytes(value) + b"\n"
                    stream.write(line)
                    h.update(line)
                    size += len(line)
            raw.flush()
            os.fsync(raw.fileno())
        os.link(name, path)
        os.unlink(name)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return dict(
        **header,
        file=path.name,
        sha256=file_hash(path),
        bytes=path.stat().st_size,
        canonical_sha256=h.hexdigest(),
        canonical_bytes=size,
    )


def read_evidence(path, metadata, identity, attempt):
    require(
        metadata["contract"] == CONTRACT
        and metadata["identity"] == identity
        and metadata["attempt"] == attempt,
        "evidence identity",
    )
    require(
        file_hash(path) == metadata["sha256"]
        and Path(path).stat().st_size == metadata["bytes"],
        "evidence bytes",
    )
    h = hashlib.sha256()
    rows = []
    size = 0
    with gzip.open(path, "rb") as stream:
        for line in stream:
            value = strict_load(line)
            require(canonical_bytes(value) + b"\n" == line, "noncanonical evidence")
            rows.append(value)
            h.update(line)
            size += len(line)
    header = rows.pop(0)
    require(
        header
        == {
            k: metadata[k] for k in ("contract", "identity", "attempt", "record_count")
        },
        "header",
    )
    require(
        len(rows) == metadata["record_count"]
        and size == metadata["canonical_bytes"]
        and h.hexdigest() == metadata["canonical_sha256"],
        "record digest/count",
    )
    validate_records(rows)
    return rows


@contextmanager
def native_journal(path, identity, attempt):
    """Durable per-step crash evidence; no restart/recovery authorization."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(
            canonical_bytes(dict(contract=CONTRACT, identity=identity, attempt=attempt))
            + b"\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
        fsync_directory(path.parent)

        def append(row):
            stream.write(canonical_bytes(row) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())

        yield append
