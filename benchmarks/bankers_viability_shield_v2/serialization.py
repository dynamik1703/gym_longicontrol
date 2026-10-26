"""BankerShield evidence contract 2: normalized canonical JSONL, atomic gzip."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from numbers import Integral, Real
from pathlib import Path

import numpy as np

CONTRACT = "bankers-evidence-v2.0"


class EvidenceError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def normalize(value):
    if is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        original = value
        value = value.item()
        if isinstance(value, np.generic):
            raise TypeError("unsupported extended numeric precision")
        require(original == value, "numeric conversion loses precision")
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        result = float(value)
        require(math.isfinite(result), "nonfinite evidence value")
        require(result == value, "numeric conversion loses precision")
        return result
    if isinstance(value, Mapping):
        require(all(isinstance(k, str) for k in value), "non-string mapping key")
        return {k: normalize(value[k]) for k in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [normalize(v) for v in value]
    raise TypeError(f"unsupported evidence type {type(value).__name__}")


def canonical_bytes(value):
    return json.dumps(
        normalize(value),
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("ascii")


def digest(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def strict_load(data):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def bad(value):
        raise EvidenceError("nonfinite JSON constant " + value)

    return json.loads(data, object_pairs_hook=pairs, parse_constant=bad)


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value) + b"\n"
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            os.link(name, path)  # exclusive publication; never overwrites evidence
            os.unlink(name)
        else:
            os.replace(name, path)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    require(path.read_bytes() == payload, "atomic JSON readback mismatch")


# Presence is necessary but not sufficient: replay checks types and all meanings.
REQUIRED = frozenset(
    (
        "step",
        "method",
        "causal_pre_state",
        "observation",
        "memory",
        "causal60_memory",
        "safe_action_mask",
        "certificates",
        "proposal_action",
        "causal60_proposal",
        "selected_index",
        "selected_certificate",
        "executed_action",
        "predicted_next_state",
        "actual_next_state",
        "signed_energy_increment_kwh",
        "actual_speed_limit_m_s",
        "speed_excess_m_s",
        "terminated",
        "truncated",
        "deadline_diagnostic",
        "inherited_comparison",
        "authority",
        "timing",
    )
)


def validate_records(rows):
    require(bool(rows), "empty evidence")
    for i, row in enumerate(rows):
        require(set(row) == REQUIRED, "missing or unknown safety fields")
        require(
            type(row["step"]) is int and row["step"] == i,
            "wrong count, missing or duplicate step",
        )
        require(len(row["observation"]) == 8, "ordinary observation dimension")
        require(
            len(row["safe_action_mask"]) == 21
            and all(type(v) is bool for v in row["safe_action_mask"]),
            "mask schema",
        )
        require(len(row["certificates"]) == 21, "certificate count")
        for field in (
            "proposal_action",
            "causal60_proposal",
            "executed_action",
            "signed_energy_increment_kwh",
            "actual_speed_limit_m_s",
            "speed_excess_m_s",
        ):
            require(type(row[field]) is float, "numeric safety field type: " + field)
        for field in ("causal_pre_state", "predicted_next_state", "actual_next_state"):
            require(
                set(row[field])
                == {
                    "position_m",
                    "velocity_m_s",
                    "acceleration_m_s2",
                    "jerk_m_s3",
                    "elapsed_time_s",
                    "total_energy_kwh",
                },
                "physical state schema",
            )
            require(
                all(type(v) is float for v in row[field].values()),
                "physical state numeric type",
            )
        require(
            type(row["terminated"]) is bool and type(row["truncated"]) is bool,
            "lifecycle schema",
        )
        require(
            row["selected_index"] is None
            or (type(row["selected_index"]) is int and 0 <= row["selected_index"] < 21),
            "selection schema",
        )
        diag = row["deadline_diagnostic"]
        require(
            set(diag) == {"state", "applicable", "completion_bound_s", "viable"},
            "deadline schema",
        )
        require(type(diag["applicable"]) is bool, "deadline applicability type")
        if diag["applicable"]:
            require(
                diag["state"] == "CALCULATED"
                and type(diag["completion_bound_s"]) is float
                and type(diag["viable"]) is bool,
                "missing calculated deadline value",
            )
        else:
            require(
                diag
                == {
                    "state": "NOT_APPLICABLE",
                    "applicable": False,
                    "completion_bound_s": None,
                    "viable": None,
                },
                "undefined inapplicable diagnostic",
            )
    normalize(rows)


def write_evidence(path, records, identity, attempt):
    rows = normalize(records)
    validate_records(rows)
    header = {
        "contract": CONTRACT,
        "identity": normalize(identity),
        "attempt": attempt,
        "record_count": len(rows),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    raw_hash = hashlib.sha256()
    raw_size = 0
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
                for value in (header, *rows):
                    line = canonical_bytes(value) + b"\n"
                    raw_hash.update(line)
                    raw_size += len(line)
                    stream.write(line)
            raw.flush()
            os.fsync(raw.fileno())
        os.link(name, path)
        os.unlink(name)
        fsync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    metadata = {
        "file": path.name,
        "sha256": file_hash(path),
        "canonical_sha256": raw_hash.hexdigest(),
        "bytes": path.stat().st_size,
        "canonical_bytes": raw_size,
        **header,
    }
    recovered = read_evidence(path, metadata, identity, attempt)
    require(canonical_bytes(recovered) == canonical_bytes(rows), "normalized readback")
    return metadata


def read_evidence(path, metadata, identity, attempt):
    require(file_hash(path) == metadata["sha256"], "incorrect evidence SHA-256")
    require(Path(path).stat().st_size == metadata["bytes"], "evidence byte count")
    require(metadata["contract"] == CONTRACT, "stale contract")
    require(metadata["identity"] == normalize(identity), "stale source/config identity")
    require(metadata["attempt"] == attempt, "wrong attempt")
    raw_hash = hashlib.sha256()
    rows = []
    header = None
    raw_size = 0
    with gzip.open(path, "rb") as stream:
        for line in stream:
            raw_hash.update(line)
            raw_size += len(line)
            value = strict_load(line)
            require(
                canonical_bytes(value) + b"\n" == line, "noncanonical payload bytes"
            )
            if header is None:
                header = value
            else:
                rows.append(value)
    require(
        header
        == {
            k: metadata[k] for k in ("contract", "identity", "attempt", "record_count")
        },
        "header schema",
    )
    require(len(rows) == metadata["record_count"], "wrong record count")
    require(raw_hash.hexdigest() == metadata["canonical_sha256"], "canonical SHA-256")
    require(raw_size == metadata["canonical_bytes"], "canonical byte count")
    validate_records(rows)
    return rows
