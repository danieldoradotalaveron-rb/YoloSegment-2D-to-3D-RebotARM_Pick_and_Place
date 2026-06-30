"""Shared ChArUco helpers for the hand-eye pipeline.

Both `capture_handeye.py` and `calibrate_handeye.py` build the board, detect it,
and estimate `cam_T_target` from a single source of truth here, so capture-time
feedback and offline calibration can never diverge.

Requires OpenCV >= 4.7 (cv2.aruco.CharucoDetector). The repo `.venv` ships 4.13.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "5x5_250": cv2.aruco.DICT_5X5_250,
}


@dataclass
class BoardSpec:
    board: Any
    dictionary: Any
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    detector: Any


def load_board(json_path: str | Path) -> BoardSpec:
    """Build a CharucoBoard from the JSON written by generate_charuco.py.

    Uses the MEASURED square/marker lengths when present (`square_length_m` /
    `marker_length_m`), which is what sets the metric translation scale.
    """
    data = json.loads(Path(json_path).read_text())
    dict_name = data["dictionary"]
    if dict_name not in DICTS:
        raise ValueError(f"Unknown dictionary {dict_name!r}; known: {list(DICTS)}")
    dictionary = cv2.aruco.getPredefinedDictionary(DICTS[dict_name])

    squares_x = int(data["squares_x"])
    squares_y = int(data["squares_y"])
    # Prefer the physically measured values; fall back to nominal.
    square_len = float(data.get("square_length_m", data.get("square_length_m_nominal")))
    marker_len = float(data.get("marker_length_m", data.get("marker_length_m_nominal")))

    board = cv2.aruco.CharucoBoard((squares_x, squares_y), square_len, marker_len, dictionary)

    # Default ArUco params are too conservative for small/oblique markers on an
    # e-ink screen (we measured 2 markers default vs 9 with these). Finer adaptive
    # thresholding, subpixel refinement and a low min-perimeter unlock detection.
    det_params = cv2.aruco.DetectorParameters()
    det_params.adaptiveThreshWinSizeMin = 3
    det_params.adaptiveThreshWinSizeMax = 43
    det_params.adaptiveThreshWinSizeStep = 4
    det_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det_params.minMarkerPerimeterRate = 0.01
    det_params.maxMarkerPerimeterRate = 4.0
    det_params.polygonalApproxAccuracyRate = 0.05
    charuco_params = cv2.aruco.CharucoParameters()
    detector = cv2.aruco.CharucoDetector(board, charuco_params, det_params)
    return BoardSpec(
        board=board,
        dictionary=dictionary,
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_m=square_len,
        marker_length_m=marker_len,
        detector=detector,
    )


def intrinsics_to_K(intr: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Build the 3x3 camera matrix K and distortion vector from a RealSense
    intrinsics dict (as produced by realsense_capture.intrinsics_to_dict)."""
    K = np.array(
        [
            [intr["fx"], 0.0, intr["ppx"]],
            [0.0, intr["fy"], intr["ppy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.asarray(intr.get("coeffs", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1, 1)
    return K, dist


@dataclass
class CharucoDetection:
    charuco_corners: np.ndarray | None
    charuco_ids: np.ndarray | None
    marker_corners: Any
    marker_ids: np.ndarray | None

    @property
    def n_corners(self) -> int:
        return 0 if self.charuco_ids is None else int(len(self.charuco_ids))


def detect(spec: BoardSpec, image_bgr: np.ndarray) -> CharucoDetection:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    ch_corners, ch_ids, m_corners, m_ids = spec.detector.detectBoard(gray)
    return CharucoDetection(ch_corners, ch_ids, m_corners, m_ids)


def estimate_pose(
    spec: BoardSpec,
    det: CharucoDetection,
    K: np.ndarray,
    dist: np.ndarray,
    min_corners: int = 6,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (rvec, tvec) of `cam_T_target` via solvePnP, or None if the view
    is too weak. rvec is a Rodrigues vector, tvec is in meters."""
    if det.charuco_ids is None or det.n_corners < min_corners:
        return None
    obj_points, img_points = spec.board.matchImagePoints(det.charuco_corners, det.charuco_ids)
    if obj_points is None or len(obj_points) < 4:
        return None
    ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    return rvec.reshape(3), tvec.reshape(3)


def draw_overlay(spec: BoardSpec, image_bgr: np.ndarray, det: CharucoDetection) -> np.ndarray:
    out = image_bgr.copy()
    if det.marker_ids is not None and len(det.marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(out, det.marker_corners, det.marker_ids)
    if det.charuco_ids is not None and det.n_corners > 0:
        cv2.aruco.drawDetectedCornersCharuco(out, det.charuco_corners, det.charuco_ids, (0, 0, 255))
    return out
