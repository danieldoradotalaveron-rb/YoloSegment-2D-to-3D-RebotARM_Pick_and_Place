"""3D visualization layer (Open3D): OBB boxes, labels, scene cloud and camera.

Presentation only: it computes no poses. The `open3d` module is not imported here;
functions that need it receive it as an `o3d` argument from the entrypoint, just like
the original script. Consumes `Detection3D` objects produced by `pipeline.py`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pipeline import Detection3D  # noqa: E402

BOX_LINES = np.array(
    [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7],
    ],
    dtype=np.int32,
)

BACKGROUND_COLOR_RGB = np.array([0.455, 0.569, 0.576], dtype=np.float64)
BACKGROUND_COLOR_RGBA = np.array([0.455, 0.569, 0.576, 1.0], dtype=np.float32)

WORKSPACE_BOX_COLOR = np.array([0.95, 0.95, 0.2], dtype=np.float64)

# Box color by temporal stability: green once the tracker confirms a steady pose,
# amber while still settling. Used only when detections carry a `stable` flag
# (tracker output); otherwise we fall back to the per-index palette.
STABLE_BOX_COLOR = np.array([0.2, 0.95, 0.35], dtype=np.float64)
UNSTABLE_BOX_COLOR = np.array([1.0, 0.75, 0.15], dtype=np.float64)


def workspace_box_corners(bounds) -> np.ndarray | None:
    """8 corners of the axis-aligned workspace box, ordered to match BOX_LINES."""
    if bounds is None:
        return None
    xlo, xhi = bounds["x"]
    ylo, yhi = bounds["y"]
    zlo, zhi = bounds["z"]
    values = [xlo, xhi, ylo, yhi, zlo, zhi]
    if not all(np.isfinite(values)):
        return None
    return np.array(
        [
            [xlo, ylo, zlo], [xhi, ylo, zlo], [xhi, yhi, zlo], [xlo, yhi, zlo],
            [xlo, ylo, zhi], [xhi, ylo, zhi], [xhi, yhi, zhi], [xlo, yhi, zhi],
        ],
        dtype=np.float64,
    )


def dim_points_outside_workspace(
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    bounds,
    dim_factor: float = 0.25,
) -> np.ndarray:
    """Darken scene points whose 3D position falls outside the workspace bounds.

    Keeps the surrounding context visible but visually subordinate, so the bright
    region of the viewer is exactly what the robot will treat as actionable.
    """
    if bounds is None or len(scene_points) == 0:
        return scene_colors
    p = scene_points
    inside = (
        (p[:, 0] >= bounds["x"][0]) & (p[:, 0] <= bounds["x"][1])
        & (p[:, 1] >= bounds["y"][0]) & (p[:, 1] <= bounds["y"][1])
        & (p[:, 2] >= bounds["z"][0]) & (p[:, 2] <= bounds["z"][1])
    )
    outside = ~inside
    if not np.any(outside):
        return scene_colors
    colors = scene_colors.copy()
    colors[outside] = colors[outside] * dim_factor
    return colors


def color_for_index(index: int) -> np.ndarray:
    palette = np.array(
        [
            [1.0, 0.35, 0.35],
            [0.35, 1.0, 0.55],
            [0.35, 0.7, 1.0],
            [1.0, 0.82, 0.35],
            [0.82, 0.35, 1.0],
            [0.35, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    return palette[index % len(palette)]


def color_for_detection(detection: Any, index: int = 0) -> np.ndarray:
    """Box color: green if the tracker marks it stable, amber if still settling.

    Falls back to the per-index palette for plain (untracked) detections, which have
    no `stable` attribute.
    """
    stable = getattr(detection, "stable", None)
    if stable is None:
        return color_for_index(index)
    return STABLE_BOX_COLOR if stable else UNSTABLE_BOX_COLOR


def highlight_object_points(
    scene_points: np.ndarray,
    scene_colors: np.ndarray,
    detections_3d: list[Detection3D],
) -> np.ndarray:
    if len(scene_points) == 0:
        return scene_colors

    colors = scene_colors.copy()
    for idx, detection in enumerate(detections_3d):
        if detection.center_xyz is None or detection.extent_xyz is None or detection.rotation_matrix is None:
            continue
        center = np.asarray(detection.center_xyz, dtype=np.float32)
        half_extent = 0.5 * np.asarray(detection.extent_xyz, dtype=np.float32)
        local = (scene_points.astype(np.float32) - center[None, :]) @ detection.rotation_matrix
        inside = np.all(np.abs(local) <= (half_extent[None, :] + 1e-4), axis=1)
        if np.any(inside):
            colors[inside] = 0.55 * colors[inside] + 0.45 * color_for_index(idx)
    return colors


def update_line_set(line_set: Any, corners_xyz: np.ndarray | None, color: np.ndarray, o3d: Any) -> None:
    if corners_xyz is None or len(corners_xyz) != 8:
        line_set.points = o3d.utility.Vector3dVector(np.empty((0, 3), dtype=np.float64))
        line_set.lines = o3d.utility.Vector2iVector(np.empty((0, 2), dtype=np.int32))
        line_set.colors = o3d.utility.Vector3dVector(np.empty((0, 3), dtype=np.float64))
        return

    line_set.points = o3d.utility.Vector3dVector(np.asarray(corners_xyz, dtype=np.float64))
    line_set.lines = o3d.utility.Vector2iVector(BOX_LINES)
    line_set.colors = o3d.utility.Vector3dVector(np.tile(color[None, :], (len(BOX_LINES), 1)))


def scene_center(points_xyz: np.ndarray) -> np.ndarray:
    if len(points_xyz) == 0:
        return np.array([0.0, 0.0, 0.5], dtype=np.float64)
    return points_xyz.mean(axis=0).astype(np.float64)


def configure_view(vis: Any, center_xyz: np.ndarray) -> None:
    view = vis.get_view_control()
    view.set_lookat(center_xyz.tolist())
    view.set_front([0.0, 0.0, -1.0])
    view.set_up([0.0, -1.0, 0.0])
    view.set_zoom(0.7)


def scene_extent(points_xyz: np.ndarray) -> float:
    if len(points_xyz) == 0:
        return 1.0
    mins = points_xyz.min(axis=0)
    maxs = points_xyz.max(axis=0)
    return float(max(np.linalg.norm(maxs - mins), 0.5))


def scene_eye(points_xyz: np.ndarray, center_xyz: np.ndarray) -> np.ndarray:
    distance = scene_extent(points_xyz) * 1.2
    return center_xyz + np.array([0.0, 0.0, -distance], dtype=np.float32)


def label_anchor(detection: Detection3D, corners: np.ndarray | None = None) -> np.ndarray:
    """Top corner of the box to float the label on. Pass the FILTERED corners so the
    anchor is as steady as the drawn box; otherwise it falls back to the (jittery)
    raw per-frame corners on the detection.
    """
    if corners is None:
        corners = detection.box_corners_xyz
    if corners is not None and len(corners) == 8:
        corners = np.asarray(corners, dtype=np.float32)
        return corners[corners[:, 1].argmin()]
    if detection.center_xyz is not None:
        return np.asarray(detection.center_xyz, dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


def _track_label_prefix(detection: Any) -> str:
    """`#<id> [STABLE|...] ` when the detection comes from the tracker, else ''."""
    track_id = getattr(detection, "track_id", None)
    if track_id is None:
        return ""
    state = "STABLE" if getattr(detection, "stable", False) else f"{getattr(detection, 'hits', 0)}/win"
    return f"#{track_id} [{state}] "


def format_detection_label(detection: Detection3D) -> str:
    prefix = _track_label_prefix(detection)
    if detection.center_xyz is None:
        return (
            f"{prefix}{detection.class_name} {detection.confidence:.2f}\n"
            f"pts={detection.point_count}"
        )

    # Prefer the temporally filtered pose when present (tracker output).
    center = getattr(detection, "center_filtered_xyz", None) or detection.center_xyz
    yaw = getattr(detection, "yaw_filtered_deg", None)
    if yaw is None:
        yaw = detection.yaw_deg
    cx, cy, cz = center
    # A symmetric footprint has no meaningful yaw
    yaw_reliable = getattr(detection, "yaw_reliable", True)
    if yaw is None:
        yaw_text = "n/a"
    elif not yaw_reliable:
        yaw_text = "n/a (sym)"
    else:
        yaw_text = f"{yaw:.1f} deg"
    return (
        f"{prefix}{detection.class_name} {detection.confidence:.2f}\n"
        f"xyz=({cx:.3f}, {cy:.3f}, {cz:.3f}) m\n"
        f"yaw={yaw_text}\n"
        f"pts={detection.point_count}"
    )


def build_legacy_point_cloud(o3d: Any, scene_points: np.ndarray, scene_colors: np.ndarray) -> Any:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(scene_points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(scene_colors.astype(np.float64))
    return pcd


def update_labels(vis: Any, detections_3d: list[Detection3D], corners_list: list | None = None) -> None:
    """Refresh the floating 3D text labels. `corners_list`, when given, is the list of
    FILTERED box corners aligned with `detections_3d`, used to keep each label anchored
    to the steady box instead of the jittery raw corners.
    """
    vis.clear_3d_labels()
    for idx, detection in enumerate(detections_3d):
        if detection.center_xyz is None or not getattr(detection, "in_workspace", True):
            continue
        corners = corners_list[idx] if corners_list is not None else None
        vis.add_3d_label(label_anchor(detection, corners), format_detection_label(detection))
