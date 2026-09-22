"""Convert the trusted, bundled sklearn 0.21 power model to portable NumPy data.

This is a maintainer tool, not runtime package code.  It deliberately accepts
only the small set of pickle globals used by the historic model.  That keeps
the one-time migration independent from an unsupported sklearn installation
and prevents arbitrary globals from being imported while unpickling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


class _LegacyObject:
    """Minimal target for state restored from retired sklearn classes."""


def _random_state_constructor(*_: Any) -> np.random.RandomState:
    return np.random.RandomState()


class _RestrictedLegacyUnpickler(pickle.Unpickler):
    """Load only the known data containers present in the bundled artifact."""

    def find_class(self, module: str, name: str) -> Any:
        multiarray = getattr(np, "_core", np.core).multiarray
        allowed: dict[tuple[str, str], Any] = {
            (
                "sklearn.neural_network.multilayer_perceptron",
                "MLPRegressor",
            ): _LegacyObject,
            (
                "sklearn.neural_network._stochastic_optimizers",
                "AdamOptimizer",
            ): _LegacyObject,
            ("numpy.random", "__RandomState_ctor"): _random_state_constructor,
            ("numpy.core.multiarray", "_reconstruct"): multiarray._reconstruct,
            ("numpy.core.multiarray", "scalar"): multiarray.scalar,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        try:
            return allowed[(module, name)]
        except KeyError as exc:
            raise pickle.UnpicklingError(
                f"Refusing unexpected pickle global {module}.{name}"
            ) from exc


def _load_legacy_model(path: Path) -> _LegacyObject:
    with path.open("rb") as stream:
        model = _RestrictedLegacyUnpickler(stream).load()
    if not isinstance(model, _LegacyObject):
        raise TypeError(f"Unexpected root object: {type(model)!r}")
    return model


def _validate_model(model: _LegacyObject) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if getattr(model, "activation", None) != "relu":
        raise ValueError("Only the expected ReLU hidden activation is supported")
    if getattr(model, "out_activation_", None) != "identity":
        raise ValueError("Only the expected identity output activation is supported")

    coefficients = [np.asarray(value, dtype=np.float64) for value in model.coefs_]
    intercepts = [np.asarray(value, dtype=np.float64) for value in model.intercepts_]
    if len(coefficients) != len(intercepts) or not coefficients:
        raise ValueError("Coefficient and intercept layer counts do not match")
    if coefficients[0].shape[0] != 2 or coefficients[-1].shape[1] != 1:
        raise ValueError("Expected two input features and one power output")
    for index, (coefficient, intercept) in enumerate(
        zip(coefficients, intercepts, strict=True)
    ):
        if coefficient.ndim != 2 or intercept.shape != (coefficient.shape[1],):
            raise ValueError(f"Invalid layer {index} shapes")
        if index and coefficients[index - 1].shape[1] != coefficient.shape[0]:
            raise ValueError(f"Layer {index} is disconnected from its predecessor")
    return coefficients, intercepts


def convert(source: Path, destination: Path, metadata_path: Path) -> None:
    model = _load_legacy_model(source)
    coefficients, intercepts = _validate_model(model)

    arrays: dict[str, np.ndarray] = {
        "hidden_activation": np.array("relu"),
        "output_activation": np.array("identity"),
    }
    for index, (coefficient, intercept) in enumerate(
        zip(coefficients, intercepts, strict=True)
    ):
        arrays[f"coef_{index}"] = coefficient
        arrays[f"intercept_{index}"] = intercept

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)

    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata = {
        "artifact_format": 1,
        "source": source.name,
        "source_sha256": source_digest,
        "source_sklearn_version": getattr(model, "_sklearn_version", "unknown"),
        "feature_order": ["velocity_m_s", "acceleration_m_s2"],
        "output": "power_kW",
        "hidden_activation": "relu",
        "output_activation": "identity",
        "layer_shapes": [list(value.shape) for value in coefficients],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=(
            repository_root
            / "legacy/v0/gym_longicontrol/envs/assets/vehicle/BMW_electric_i3_2014.pkl"
        ),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=(
            repository_root
            / "src/gym_longicontrol/assets/vehicle/BMW_electric_i3_2014.npz"
        ),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=(
            repository_root
            / "src/gym_longicontrol/assets/vehicle/BMW_electric_i3_2014.json"
        ),
    )
    arguments = parser.parse_args()
    convert(arguments.source, arguments.destination, arguments.metadata)


if __name__ == "__main__":
    main()
