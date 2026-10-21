"""Physical time-price units and the frozen normalized action mapping."""

from __future__ import annotations

import math

import numpy as np

# Provisional physical bounds are frozen by ``canonical.json`` after the
# Development calibration.  They span the observed marginal prices without a
# zero-valued logarithm singularity.
LAMBDA_MIN_KWH_PER_S = 1.0e-5
LAMBDA_MAX_KWH_PER_S = 1.0e-2


def normalized_action_to_lambda(action: float) -> float:
    """Map [-1, 1] monotonically onto a positive logarithmic price interval."""

    value = float(action)
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError("normalized time-price action must be finite in [-1, 1]")
    if value == -1.0:
        return LAMBDA_MIN_KWH_PER_S
    if value == 1.0:
        return LAMBDA_MAX_KWH_PER_S
    fraction = 0.5 * (value + 1.0)
    log_value = math.log(LAMBDA_MIN_KWH_PER_S) + fraction * math.log(
        LAMBDA_MAX_KWH_PER_S / LAMBDA_MIN_KWH_PER_S
    )
    return math.exp(log_value)


def lambda_to_normalized_action(price_kwh_per_s: float) -> float:
    """Invert :func:`normalized_action_to_lambda`, clipping to frozen bounds."""

    value = float(price_kwh_per_s)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("time price must be finite and nonnegative")
    clipped = float(np.clip(value, LAMBDA_MIN_KWH_PER_S, LAMBDA_MAX_KWH_PER_S))
    fraction = math.log(clipped / LAMBDA_MIN_KWH_PER_S) / math.log(
        LAMBDA_MAX_KWH_PER_S / LAMBDA_MIN_KWH_PER_S
    )
    return 2.0 * fraction - 1.0


def marginal_price(
    *,
    slow_energy_kwh: float,
    fast_energy_kwh: float,
    slow_time_s: float,
    fast_time_s: float,
) -> float:
    """Return the positive finite-difference price between two trade-off points."""

    delta_time = float(slow_time_s) - float(fast_time_s)
    if not math.isfinite(delta_time) or delta_time <= 0.0:
        raise ValueError("slow trajectory must take strictly longer")
    delta_energy = float(fast_energy_kwh) - float(slow_energy_kwh)
    if not math.isfinite(delta_energy):
        raise ValueError("energies must be finite")
    return max(0.0, delta_energy / delta_time)
