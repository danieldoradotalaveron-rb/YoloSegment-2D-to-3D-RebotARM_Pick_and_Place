"""Consume the hand-eye calibration: map points from the camera frame to the
robot base frame.
"""

from __future__ import annotations

import numpy as np


def cam_point_to_base(
    cam_xyz: np.ndarray,
    base_T_ee: np.ndarray,
    ee_T_cam: np.ndarray,
) -> np.ndarray:
    """Map a camera-frame point to the robot base frame (eye-in-hand).

        base_point = base_T_ee @ ee_T_cam @ [x, y, z, 1]

    cam_xyz: (3,) point in camera frame, meters.
    base_T_ee: (4, 4) end-effector pose in base, from TF, per frame.
    ee_T_cam: (4, 4) camera pose in end-effector, hand-eye result.
    Returns the (3,) point in base frame, meters.
    """
    cam_xyz = np.asarray(cam_xyz, dtype=np.float64).reshape(3)
    base_T_ee = np.asarray(base_T_ee, dtype=np.float64)
    ee_T_cam = np.asarray(ee_T_cam, dtype=np.float64)
    point_h = np.array([cam_xyz[0], cam_xyz[1], cam_xyz[2], 1.0], dtype=np.float64)
    base_h = base_T_ee @ ee_T_cam @ point_h
    return base_h[:3]
