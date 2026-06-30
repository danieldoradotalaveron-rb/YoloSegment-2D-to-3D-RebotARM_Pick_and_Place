"""Circular statistics for tabletop box yaw stabilization.
"""

from __future__ import annotations

import numpy as np


def _doubled(yaws_deg: list[float]) -> np.ndarray:
    """Yaws (degrees) as doubled angles in radians
    """
    return np.radians(np.asarray(yaws_deg, dtype=np.float64) * 2.0)


def circular_mean_deg(yaws_deg: list[float]) -> float:
    """Average yaw (degrees)
    """
    if not yaws_deg:
        return 0.0
    doubled = _doubled(yaws_deg)
    mean = np.arctan2(np.sin(doubled).mean(), np.cos(doubled).mean())
    return float(np.degrees(mean) / 2.0)


def circular_std_deg(yaws_deg: list[float]) -> float:
    """Circular standard deviation of yaw (degrees), same 180-degree convention.
    """
    if len(yaws_deg) < 2:
        return 0.0
    doubled = _doubled(yaws_deg)
    resultant = min(1.0, max(0.0, float(np.hypot(np.sin(doubled).mean(), np.cos(doubled).mean()))))
    if resultant <= 1e-9:
        return 90.0
    return float(np.degrees(np.sqrt(-2.0 * np.log(resultant))) / 2.0)
