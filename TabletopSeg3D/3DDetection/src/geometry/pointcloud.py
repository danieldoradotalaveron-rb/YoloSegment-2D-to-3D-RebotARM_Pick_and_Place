from __future__ import annotations

from typing import Any

import numpy as np


def _valid_mask(
    mask: np.ndarray,
    depth_m: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    """Pixels that are inside the mask and have a usable depth value."""
    return (
        mask
        & np.isfinite(depth_m)
        & (depth_m > min_depth_m)
        & (depth_m < max_depth_m)
    )


def _backproject_pixels(
    ys: np.ndarray,
    xs: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict[str, Any],
) -> np.ndarray:
    """Pinhole backprojection of (ys, xs) pixels to camera-frame XYZ (meters)."""
    z = depth_m[ys, xs].astype(np.float32)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    ppx = float(intrinsics["ppx"])
    ppy = float(intrinsics["ppy"])

    x = (xs.astype(np.float32) - ppx) * z / fx
    y = (ys.astype(np.float32) - ppy) * z / fy
    return np.stack([x, y, z], axis=1)


def _lift_mask_pixels(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict[str, Any],
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single source of truth: validity + pixel indices + XYZ.

    Returns (points_xyz, ys, xs, valid_mask). On no valid pixels, points_xyz is
    an empty (0, 3) array and ys/xs are empty index arrays.
    """
    if depth_m.shape[:2] != mask.shape[:2]:
        raise RuntimeError(
            f"Depth shape {depth_m.shape[:2]} does not match mask shape {mask.shape[:2]}."
        )

    valid_mask = _valid_mask(mask, depth_m, min_depth_m, max_depth_m)
    ys, xs = np.where(valid_mask)
    if len(xs) == 0:
        return np.empty((0, 3), dtype=np.float32), ys, xs, valid_mask

    points = _backproject_pixels(ys, xs, depth_m, intrinsics)
    return points, ys, xs, valid_mask


def project_mask_to_points(
    mask: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict[str, Any],
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    points, _, _, valid_mask = _lift_mask_pixels(
        mask, depth_m, intrinsics, min_depth_m, max_depth_m
    )
    return points, valid_mask


def project_mask_to_colored_points(
    mask: np.ndarray,
    depth_m: np.ndarray,
    rgb: np.ndarray,
    intrinsics: dict[str, Any],
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lift a mask to camera-frame points carrying per-pixel color and (u, v).

    Shares the exact backprojection of `project_mask_to_points`. Intended for
    offline 3DGS seeding (not used by the realtime tabletop viewer).

    Returns:
        points_xyz: (N, 3) float32, camera frame, meters.
        colors:     (N, 3) matching rgb.dtype/channel-order (BGR if from OpenCV).
        pixel_uv:   (N, 2) int32, columns are (u=x, v=y) image coordinates.
    """
    if rgb.shape[:2] != mask.shape[:2]:
        raise RuntimeError(
            f"RGB shape {rgb.shape[:2]} does not match mask shape {mask.shape[:2]}."
        )

    points, ys, xs, _ = _lift_mask_pixels(
        mask, depth_m, intrinsics, min_depth_m, max_depth_m
    )
    if len(xs) == 0:
        colors = np.empty((0, rgb.shape[2]) if rgb.ndim == 3 else (0,), dtype=rgb.dtype)
        return points, colors, np.empty((0, 2), dtype=np.int32)

    colors = rgb[ys, xs]
    pixel_uv = np.stack([xs, ys], axis=1).astype(np.int32)
    return points, colors, pixel_uv


def filter_points_by_depth_band(points_xyz: np.ndarray, z_mad_scale: float = 2.5) -> np.ndarray:
    if len(points_xyz) == 0:
        return points_xyz

    z = points_xyz[:, 2]
    median = np.median(z)
    mad = np.median(np.abs(z - median))
    if mad < 1e-6:
        return points_xyz

    keep = np.abs(z - median) <= z_mad_scale * 1.4826 * mad
    filtered = points_xyz[keep]
    return filtered if len(filtered) > 0 else points_xyz


def _normalize_vector(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return fallback.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _project_vector_to_plane(vector: np.ndarray, plane_normal: np.ndarray) -> np.ndarray:
    return vector - np.dot(vector, plane_normal) * plane_normal


def _wrap_half_turn(angle_rad: float) -> float:
    return float((angle_rad + (0.5 * np.pi)) % np.pi - (0.5 * np.pi))


def plane_basis_from_normal(
    plane_normal: np.ndarray,
    reference_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fallback_normal = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    normal = _normalize_vector(np.asarray(plane_normal, dtype=np.float32), fallback_normal)

    ref = np.array([1.0, 0.0, 0.0], dtype=np.float32) if reference_axis is None else np.asarray(reference_axis, dtype=np.float32)
    axis_u = _project_vector_to_plane(ref, normal)
    if np.linalg.norm(axis_u) < 1e-6:
        axis_u = _project_vector_to_plane(np.array([0.0, 0.0, 1.0], dtype=np.float32), normal)
    if np.linalg.norm(axis_u) < 1e-6:
        axis_u = _project_vector_to_plane(np.array([0.0, 1.0, 0.0], dtype=np.float32), normal)
    axis_u = _normalize_vector(axis_u, np.array([1.0, 0.0, 0.0], dtype=np.float32))

    axis_v = np.cross(normal, axis_u)
    axis_v = _normalize_vector(axis_v, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    axis_u = _normalize_vector(np.cross(axis_v, normal), axis_u)
    return axis_u, axis_v, normal


def obb_corners_from_pose(
    center_xyz: np.ndarray,
    extent_xyz: np.ndarray,
    yaw_rad: float,
    plane_normal: np.ndarray,
    reference_axis: np.ndarray | None = None,
) -> np.ndarray:
    """Rebuild the 8 OBB corners from a tabletop-aligned pose.

    Inverse of the corner emission in `tabletop_aligned_obb`: given a box center,
    extent and yaw (all in the same table frame derived from `plane_normal`), it
    reconstructs the world-frame corners. Used to draw a box from the temporally
    FILTERED pose so the viewer matches the reported (smoothed) numbers, instead of
    the raw per-frame corners. Corner order matches BOX_LINES in viz_3d.
    """
    base_u, base_v, normal = plane_basis_from_normal(plane_normal, reference_axis=reference_axis)
    cos_yaw = float(np.cos(yaw_rad))
    sin_yaw = float(np.sin(yaw_rad))
    axis_long = _normalize_vector(cos_yaw * base_u + sin_yaw * base_v, base_u)
    axis_short = _normalize_vector(-sin_yaw * base_u + cos_yaw * base_v, base_v)
    rotation = np.stack([axis_long, axis_short, normal], axis=1).astype(np.float32)

    hx, hy, hz = (np.asarray(extent_xyz, dtype=np.float32) * 0.5).tolist()
    corners_local = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    center = np.asarray(center_xyz, dtype=np.float32)
    return (center[None, :] + corners_local @ rotation.T).astype(np.float32)


def tabletop_aligned_obb(
    points_xyz: np.ndarray,
    plane_normal: np.ndarray,
    reference_axis: np.ndarray | None = None,
    force_yaw_zero: bool = False,
) -> dict[str, Any]:
    """Tabletop-aligned oriented bounding box.

    With ``force_yaw_zero`` the box is axis-aligned to the table frame (base_u/base_v)
    instead of the PCA principal axis: used for symmetric objects (no meaningful yaw),
    so the drawn box matches the frozen yaw=0 and stays tight instead of mis-rotated.
    """
    if len(points_xyz) == 0:
        zeros = np.zeros(3, dtype=np.float32)
        return {
            "center_xyz": zeros,
            "extent_xyz": zeros,
            "rotation_matrix": np.eye(3, dtype=np.float32),
            "corners_xyz": np.zeros((8, 3), dtype=np.float32),
            "yaw_rad": 0.0,
            "yaw_deg": 0.0,
            "bbox_min_xyz": zeros,
            "bbox_max_xyz": zeros,
        }

    base_u, base_v, normal = plane_basis_from_normal(plane_normal, reference_axis=reference_axis)
    planar = np.stack([points_xyz @ base_u, points_xyz @ base_v], axis=1)
    planar_centered = planar - planar.mean(axis=0, keepdims=True)

    if force_yaw_zero:
        # Table-axis-aligned (AABB in the plane): no PCA, yaw is meaningless here.
        major_axis_2d = np.array([1.0, 0.0], dtype=np.float32)
    elif len(points_xyz) >= 3:
        covariance = np.cov(planar_centered, rowvar=False)
        if covariance.shape == (2, 2) and np.all(np.isfinite(covariance)):
            eigvals, eigvecs = np.linalg.eigh(covariance)
            major_axis_2d = eigvecs[:, int(np.argmax(eigvals))]
        else:
            major_axis_2d = np.array([1.0, 0.0], dtype=np.float32)
    else:
        major_axis_2d = np.array([1.0, 0.0], dtype=np.float32)

    yaw_rad = _wrap_half_turn(float(np.arctan2(major_axis_2d[1], major_axis_2d[0])))
    cos_yaw = float(np.cos(yaw_rad))
    sin_yaw = float(np.sin(yaw_rad))

    axis_long = _normalize_vector(cos_yaw * base_u + sin_yaw * base_v, base_u)
    axis_short = _normalize_vector(-sin_yaw * base_u + cos_yaw * base_v, base_v)
    rotation = np.stack([axis_long, axis_short, normal], axis=1).astype(np.float32)

    local_xyz = np.stack(
        [
            points_xyz @ axis_long,
            points_xyz @ axis_short,
            points_xyz @ normal,
        ],
        axis=1,
    )
    mins = local_xyz.min(axis=0)
    maxs = local_xyz.max(axis=0)
    extent = (maxs - mins).astype(np.float32)
    center_local = ((mins + maxs) * 0.5).astype(np.float32)
    center_world = (rotation @ center_local).astype(np.float32)

    hx, hy, hz = (extent * 0.5).tolist()
    corners_local = np.array(
        [
            [-hx, -hy, -hz],
            [hx, -hy, -hz],
            [hx, hy, -hz],
            [-hx, hy, -hz],
            [-hx, -hy, hz],
            [hx, -hy, hz],
            [hx, hy, hz],
            [-hx, hy, hz],
        ],
        dtype=np.float32,
    )
    corners_world = center_world[None, :] + corners_local @ rotation.T
    bbox_min = corners_world.min(axis=0).astype(np.float32)
    bbox_max = corners_world.max(axis=0).astype(np.float32)

    return {
        "center_xyz": center_world,
        "extent_xyz": extent,
        "rotation_matrix": rotation,
        "corners_xyz": corners_world.astype(np.float32),
        "yaw_rad": float(yaw_rad),
        "yaw_deg": float(np.degrees(yaw_rad)),
        "bbox_min_xyz": bbox_min,
        "bbox_max_xyz": bbox_max,
    }
