from __future__ import annotations

import json
from importlib.resources import files

import numpy as np

from gym_longicontrol.domain.vehicle import NumpyPowerModel


def _artifact_forward(features: np.ndarray) -> np.ndarray:
    return NumpyPowerModel().predict(features)


def test_portable_power_model_matches_legacy_golden_predictions() -> None:
    features = np.array(
        [
            [0.0, 0.0],
            [5.0, 1.0],
            [10.0, 0.0],
            [20.0, -1.0],
            [30.0, 2.0],
            [37.0, -3.0],
        ]
    )
    expected_power_kw = np.array(
        [
            0.80876967,
            10.69562794,
            4.76220620,
            -16.33766137,
            99.75707891,
            -64.08281322,
        ]
    )

    actual = _artifact_forward(features).reshape(-1)

    np.testing.assert_allclose(actual, expected_power_kw, rtol=0.0, atol=5e-9)


def test_power_model_metadata_documents_provenance_and_units() -> None:
    resource = files("gym_longicontrol").joinpath(
        "assets", "vehicle", "BMW_electric_i3_2014.json"
    )
    metadata = json.loads(resource.read_text(encoding="utf-8"))

    assert metadata["artifact_format"] == 1
    assert metadata["source_sklearn_version"] == "0.21.2"
    assert metadata["feature_order"] == ["velocity_m_s", "acceleration_m_s2"]
    assert metadata["output"] == "power_kW"
    assert metadata["layer_shapes"] == [[2, 10], [10, 10], [10, 10], [10, 1]]
    assert len(metadata["source_sha256"]) == 64
