"""Virtual pinhole cameras for synthetic view rendering (camera frame)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def intrinsics_matrix(intrinsics: dict[str, Any]) -> np.ndarray:
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    ppx = float(intrinsics["ppx"])
    ppy = float(intrinsics["ppy"])
    return np.array([[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_yaw_pitch(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Rotate scene points: pitch around X, then yaw around Y (camera frame)."""
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cx, sx = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
    return ry @ rx


def default_virtual_views() -> list[dict[str, Any]]:
    """Reference + small offsets (degrees) per Phase 1B plan."""
    specs = [
        ("view_000", 0.0, 0.0),
        ("view_001", 15.0, 0.0),
        ("view_002", -15.0, 0.0),
        ("view_003", 0.0, 10.0),
        ("view_004", 0.0, -10.0),
    ]
    views = []
    for view_id, yaw, pitch in specs:
        views.append(
            {
                "view_id": view_id,
                "yaw_deg": yaw,
                "pitch_deg": pitch,
                "rotation": rotation_yaw_pitch(yaw, pitch),
            }
        )
    return views


def project_points(
    xyz: np.ndarray,
    intrinsics: dict[str, Any],
    rotation: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project camera-frame points to pixel coords. Returns (u, v, z) for valid z>0."""
    pts = xyz.astype(np.float64)
    if rotation is not None:
        pts = (rotation @ pts.T).T
    z = pts[:, 2]
    valid = z > 1e-3
    k = intrinsics_matrix(intrinsics)
    uvw = (k @ pts.T).T
    u = uvw[:, 0] / np.maximum(z, 1e-6)
    v = uvw[:, 1] / np.maximum(z, 1e-6)
    return u, v, z
