"""Complete immutable strategic-boundary snapshots and exact restoration."""

from __future__ import annotations

import copy
import gzip
import hashlib
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np

from .instrumented_controller import InstrumentedContinueController

SNAPSHOT_SCHEMA_VERSION = 1


def _space_rng_state(space: Any) -> Any:
    generator = getattr(space, "_np_random", None)
    return None if generator is None else copy.deepcopy(generator.bit_generator.state)


def capture_snapshot(
    *,
    environment: Any,
    controller: InstrumentedContinueController,
    observation: np.ndarray,
    info: dict[str, Any],
    held_price: float | None,
    native_step: int,
    selector_last_action: float,
) -> dict[str, Any]:
    unwrapped = environment.unwrapped
    return {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "environment_pickle": pickle.dumps(environment, protocol=5),
        "environment_rng_state": copy.deepcopy(unwrapped.np_random.bit_generator.state),
        "action_space_rng_state": _space_rng_state(environment.action_space),
        "observation_space_rng_state": _space_rng_state(environment.observation_space),
        "controller": controller.snapshot(),
        "observation": np.asarray(observation, dtype=np.float64),
        "info": copy.deepcopy(info),
        "held_price": held_price,
        "strategic_decision_clock": int(native_step % 10),
        "native_step": int(native_step),
        "selector_last_action": float(selector_last_action),
        "python_random_state": random.getstate(),
        "numpy_legacy_rng_state": np.random.get_state(),
        "pending_observations_actions": {
            "controller_pending_observation": controller.snapshot()[
                "pending_observation"
            ],
            "controller_pending_fallback": controller.snapshot()["pending_fallback"],
        },
    }


def snapshot_bytes(payload: dict[str, Any]) -> bytes:
    return gzip.compress(pickle.dumps(payload, protocol=5), compresslevel=6, mtime=0)


def write_snapshot(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = snapshot_bytes(payload)
    destination.write_bytes(encoded)
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
    }


def read_snapshot(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    encoded = Path(path).read_bytes()
    observed = hashlib.sha256(encoded).hexdigest()
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError("snapshot hash mismatch")
    payload = pickle.loads(gzip.decompress(encoded))
    if payload.get("snapshot_schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported snapshot schema")
    return payload


def restore_snapshot(
    payload: dict[str, Any],
) -> tuple[
    Any,
    InstrumentedContinueController,
    np.ndarray,
    dict[str, Any],
    float | None,
    int,
    float,
]:
    if payload.get("snapshot_schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError("unsupported snapshot schema")
    environment = pickle.loads(payload["environment_pickle"])
    controller = InstrumentedContinueController()
    controller.restore(payload["controller"])
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_legacy_rng_state"])
    if (
        environment.unwrapped.np_random.bit_generator.state
        != payload["environment_rng_state"]
    ):
        raise RuntimeError("environment RNG did not restore exactly")
    for key, space in (
        ("action_space_rng_state", environment.action_space),
        ("observation_space_rng_state", environment.observation_space),
    ):
        if _space_rng_state(space) != payload[key]:
            raise RuntimeError(f"{key} did not restore exactly")
    return (
        environment,
        controller,
        np.asarray(payload["observation"], dtype=np.float64),
        copy.deepcopy(payload["info"]),
        payload["held_price"],
        int(payload["native_step"]),
        float(payload["selector_last_action"]),
    )
