"""Vehicle constraints and portable inference of the historical power MLP."""

from dataclasses import dataclass
from importlib.resources import files
from typing import Protocol

import numpy as np


class PowerModel(Protocol):
    def predict(self, features: np.ndarray) -> np.ndarray: ...


class NumpyPowerModel:
    def __init__(self, resource=None):
        if resource is None:
            resource = files("gym_longicontrol").joinpath(
                "assets/vehicle/BMW_electric_i3_2014.npz"
            )
        with resource.open("rb") as stream, np.load(stream, allow_pickle=False) as data:
            if str(data["hidden_activation"]) != "relu":
                raise ValueError("Unsupported hidden activation")
            if str(data["output_activation"]) != "identity":
                raise ValueError("Unsupported output activation")
            count = sum(key.startswith("coef_") for key in data.files)
            expected = {"hidden_activation", "output_activation"}
            expected.update(f"coef_{i}" for i in range(count))
            expected.update(f"intercept_{i}" for i in range(count))
            if count == 0 or set(data.files) != expected:
                raise ValueError("Invalid power-model layer keys")
            self.layers = tuple(
                (
                    np.array(data[f"coef_{i}"], dtype=np.float64),
                    np.array(data[f"intercept_{i}"], dtype=np.float64),
                )
                for i in range(count)
            )
        width = 2
        for weight, bias in self.layers:
            if (
                weight.ndim != 2
                or weight.shape[0] != width
                or bias.shape != (weight.shape[1],)
                or not np.isfinite(weight).all()
                or not np.isfinite(bias).all()
            ):
                raise ValueError("Invalid power-model layer shapes or values")
            width = weight.shape[1]
            weight.setflags(write=False)
            bias.setflags(write=False)
        if width != 1:
            raise ValueError("Power model must have one output")

    def predict(self, features):
        value = np.asarray(features, dtype=np.float64)
        if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
            raise ValueError("Power features must be a finite (n, 2) array")
        for index, (weight, bias) in enumerate(self.layers):
            value = value @ weight + bias
            if index + 1 < len(self.layers):
                value = np.maximum(value, 0.0)
        return value[:, 0]


@dataclass(frozen=True)
class VehicleSpecs:
    mass_kg: float = 1443.0
    frontal_area_m2: float = 2.38
    drag_coefficient: float = 0.29
    acceleration_limits: tuple[float, float] = (-3.0, 3.0)
    velocity_limits: tuple[float, float] = (0.0, 37.0)
    power_limits_kw: tuple[float, float] = (-50.0, 75.0)


class VehicleModel:
    def __init__(self, car_id="BMW_electric_i3_2014", power_model=None):
        if car_id != "BMW_electric_i3_2014":
            raise ValueError(
                f"Unknown car_id {car_id!r}; only BMW_electric_i3_2014 exists"
            )
        self.specs = VehicleSpecs()
        self.power_model = power_model if power_model is not None else NumpyPowerModel()
        velocity = np.arange(1, self.specs.velocity_limits[1] + 1)
        self.velocity_grid = np.r_[0.0, velocity]
        self.positive_acceleration = np.r_[
            1.0, np.minimum(self.acceleration_from_power(velocity, 75.0), 3.0)
        ]
        self.negative_acceleration = np.r_[
            0.0, np.maximum(self.acceleration_from_power(velocity, -50.0), -3.0)
        ]
        self.coasting_acceleration = np.r_[
            0.0, self.acceleration_from_power(velocity, 0.0)
        ]

    def acceleration_from_power(self, velocity, power_kw):
        specs = self.specs
        return power_kw * 1000 / (velocity * specs.mass_kg) - (
            specs.drag_coefficient
            * specs.frontal_area_m2
            * 1.2
            * velocity**2
            / (2 * specs.mass_kg)
            + 0.015 * 9.81
        )

    def acceleration_from_action(self, velocity_m_s: float, action: float) -> float:
        minimum = np.interp(
            velocity_m_s,
            self.velocity_grid,
            self.negative_acceleration,
            left=0,
            right=-1e-6,
        )
        coast = np.interp(
            velocity_m_s,
            self.velocity_grid,
            self.coasting_acceleration,
            left=0,
            right=-1e-6,
        )
        maximum = np.interp(
            velocity_m_s,
            self.velocity_grid,
            self.positive_acceleration,
            left=0.6258544444444445,
            right=-1e-6,
        )
        limit = maximum if action > 0 else minimum
        return float((limit - coast) * abs(action) + coast)

    def power_kw(self, velocity_m_s: float, acceleration_m_s2: float) -> float:
        value = float(
            self.power_model.predict(np.array([[velocity_m_s, acceleration_m_s2]]))[0]
        )
        if not np.isfinite(value):
            raise ValueError("Vehicle power model returned non-finite power")
        return value
