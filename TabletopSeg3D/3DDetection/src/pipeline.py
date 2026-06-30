"""3D detection core: RGB-D bundle -> 3D detections -> JSON record.

No GUI dependencies. Only needs numpy and `geometry.pointcloud`. `cv2` is used
solely to resize the YOLO mask to the image resolution, and `open3d` is not
imported here: the table-plane RANSAC receives the `o3d` module as an argument.
This is the module the robot can import to obtain poses without opening any window.

Flow: get_aligned_frame_bundle() -> run_inference() -> build_detection_3d()
      -> frame_output_record() (JSON).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np

from geometry.pointcloud import (  # noqa: E402
    filter_points_by_depth_band,
    obb_corners_from_pose,
    project_mask_to_points,
    tabletop_aligned_obb,
)


@dataclass
class Detection3D:
    class_name: str
    confidence: float
    bbox_xyxy: list[int]
    mask: np.ndarray
    center_xyz: list[float] | None
    extent_xyz: list[float] | None
    yaw_rad: float | None
    yaw_deg: float | None
    rotation_matrix: np.ndarray | None
    box_corners_xyz: np.ndarray | None
    bbox_min_xyz: list[float] | None
    bbox_max_xyz: list[float] | None
    point_count: int
    in_workspace: bool = True
    # Whether this class has a meaningful orientation. Declared by class policy
    # (only `non_symmetric_classes` get a trusted yaw); symmetric objects
    # (cork, pompom, ...) are grasped top-down and their yaw is treated as free.
    yaw_reliable: bool = False


def parse_workspace(value: str | None):
    """Parse 'xmin,xmax,ymin,ymax,zmin,zmax' (meters, camera frame) -> bounds dict.

    Returns None when empty/invalid (workspace disabled). The same bounds drive both
    the 3D viewer box and the actionable filtering, so the view matches the robot.
    """
    if not value or not str(value).strip():
        return None
    try:
        parts = [float(p) for p in str(value).replace(" ", "").split(",")]
    except ValueError:
        return None
    if len(parts) != 6:
        return None
    xlo, xhi, ylo, yhi, zlo, zhi = parts
    return {
        "x": (min(xlo, xhi), max(xlo, xhi)),
        "y": (min(ylo, yhi), max(ylo, yhi)),
        "z": (min(zlo, zhi), max(zlo, zhi)),
    }


def parse_class_set(value: str | None) -> set[str]:
    """Parse a comma-separated class list ('cork,pompom') into a set of names."""
    if not value or not str(value).strip():
        return set()
    return {name.strip() for name in str(value).split(",") if name.strip()}


def center_in_workspace(center, bounds) -> bool:
    """True if a 3D point (camera frame) lies inside the workspace bounds.

    With ``bounds`` None (workspace disabled) everything is considered actionable.
    """
    if bounds is None or center is None:
        return True
    x, y, z = (float(center[0]), float(center[1]), float(center[2]))
    return (
        bounds["x"][0] <= x <= bounds["x"][1]
        and bounds["y"][0] <= y <= bounds["y"][1]
        and bounds["z"][0] <= z <= bounds["z"][1]
    )


def load_model(model_name: str):
    from ultralytics import YOLO

    return YOLO(model_name)


def dedup_detections_by_mask_iou(
    detections: list[dict[str, Any]], iou_thr: float
) -> list[dict[str, Any]]:
    """Drop same-class detections overlapping a higher-confidence one above ``iou_thr``.

    Box NMS (inside predict) keeps two near-identical masks when their *boxes* fall
    below its IoU threshold, which then become two 3D boxes on one object. This
    collapses those by comparing the actual instance masks. Keeps the original order.
    """
    order = sorted(range(len(detections)), key=lambda k: detections[k]["confidence"], reverse=True)
    kept: list[int] = []
    for k in order:
        mk = detections[k]["mask"]
        is_dup = False
        for j in kept:
            if detections[j]["class_name"] != detections[k]["class_name"]:
                continue
            mj = detections[j]["mask"]
            inter = int(np.logical_and(mk, mj).sum())
            if inter == 0:
                continue
            union = int(np.logical_or(mk, mj).sum())
            if union and inter / union >= iou_thr:
                is_dup = True
                break
        if not is_dup:
            kept.append(k)
    return [detections[k] for k in sorted(kept)]


def run_inference(model, color_image: np.ndarray, args: argparse.Namespace) -> list[dict[str, Any]]:
    import cv2

    results = model.predict(
        source=color_image,
        task="segment",
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        verbose=False,
    )
    result = results[0]
    if result.masks is None or result.boxes is None:
        return []

    masks = result.masks.data.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy().astype(float)
    bboxes = result.boxes.xyxy.cpu().numpy().astype(int)
    image_h, image_w = color_image.shape[:2]

    detections: list[dict[str, Any]] = []
    for idx, mask_arr in enumerate(masks):
        class_id = int(class_ids[idx])
        class_name = result.names.get(class_id, str(class_id))
        if args.target_class and class_name != args.target_class:
            continue
        mask_resized = cv2.resize(mask_arr, (image_w, image_h), interpolation=cv2.INTER_NEAREST) > 0.5
        detections.append(
            {
                "class_name": class_name,
                "confidence": float(confidences[idx]),
                "bbox_xyxy": [int(v) for v in bboxes[idx].tolist()],
                "mask": mask_resized,
            }
        )

    dedup_iou = getattr(args, "dedup_iou", 0.0)
    if dedup_iou and dedup_iou > 0 and len(detections) > 1:
        detections = dedup_detections_by_mask_iou(detections, dedup_iou)
    return detections


def build_detection_3d(
    detection: dict[str, Any],
    depth_m: np.ndarray,
    intrinsics: dict[str, Any],
    table_normal: np.ndarray,
    args: argparse.Namespace,
) -> Detection3D:
    raw_points, _ = project_mask_to_points(
        mask=detection["mask"],
        depth_m=depth_m,
        intrinsics=intrinsics,
        min_depth_m=args.min_depth,
        max_depth_m=args.max_depth,
    )
    filtered_points = filter_points_by_depth_band(raw_points)
    point_count = int(len(filtered_points))
    if point_count < args.min_points:
        return Detection3D(
            class_name=detection["class_name"],
            confidence=detection["confidence"],
            bbox_xyxy=detection["bbox_xyxy"],
            mask=detection["mask"],
            center_xyz=None,
            extent_xyz=None,
            yaw_rad=None,
            yaw_deg=None,
            rotation_matrix=None,
            box_corners_xyz=None,
            bbox_min_xyz=None,
            bbox_max_xyz=None,
            point_count=point_count,
        )

    # Class policy: yaw is meaningful (and processed downstream) only for classes
    # explicitly declared non-symmetric. Everything else is treated as symmetric, and
    # we build a table-axis-aligned box (force_yaw_zero) so it stays tight and matches
    # the frozen yaw=0 used downstream.
    non_symmetric = getattr(args, "non_symmetric_set", None) or set()
    yaw_reliable = detection["class_name"] in non_symmetric
    obb = tabletop_aligned_obb(
        filtered_points, plane_normal=table_normal, force_yaw_zero=not yaw_reliable
    )
    center_xyz = obb["center_xyz"].tolist()
    return Detection3D(
        class_name=detection["class_name"],
        confidence=detection["confidence"],
        bbox_xyxy=detection["bbox_xyxy"],
        mask=detection["mask"],
        center_xyz=center_xyz,
        extent_xyz=obb["extent_xyz"].tolist(),
        yaw_rad=obb["yaw_rad"],
        yaw_deg=obb["yaw_deg"],
        rotation_matrix=obb["rotation_matrix"],
        box_corners_xyz=obb["corners_xyz"],
        bbox_min_xyz=obb["bbox_min_xyz"].tolist(),
        bbox_max_xyz=obb["bbox_max_xyz"].tolist(),
        point_count=point_count,
        in_workspace=center_in_workspace(center_xyz, getattr(args, "workspace_bounds", None)),
        yaw_reliable=yaw_reliable,
    )


def build_scene_point_cloud(
    color_image: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth_m.shape
    stride = max(1, int(args.point_stride))
    ys = np.arange(0, height, stride, dtype=np.int32)
    xs = np.arange(0, width, stride, dtype=np.int32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    sampled_depth = depth_m[grid_y, grid_x]
    valid = np.isfinite(sampled_depth) & (sampled_depth > args.min_depth) & (sampled_depth < args.max_depth)
    if not np.any(valid):
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float64)

    z = sampled_depth[valid].astype(np.float32)
    u = grid_x[valid].astype(np.float32)
    v = grid_y[valid].astype(np.float32)
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    ppx = float(intrinsics["ppx"])
    ppy = float(intrinsics["ppy"])

    x = (u - ppx) * z / fx
    y = (v - ppy) * z / fy
    points = np.stack([x, y, z], axis=1)
    colors = color_image[grid_y[valid], grid_x[valid]][:, ::-1].astype(np.float64) / 255.0

    max_points = int(args.scene_max_points)
    if max_points > 0 and len(points) > max_points:
        keep = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
        points = points[keep]
        colors = colors[keep]

    return points, colors


def estimate_table_normal(scene_points: np.ndarray, o3d: Any) -> np.ndarray:
    default_normal = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    if len(scene_points) < 128:
        return default_normal

    sampled_points = scene_points
    max_plane_points = 12000
    if len(sampled_points) > max_plane_points:
        keep = np.linspace(0, len(sampled_points) - 1, max_plane_points, dtype=np.int32)
        sampled_points = sampled_points[keep]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sampled_points.astype(np.float64))
    plane_model, _ = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=120)
    normal = np.asarray(plane_model[:3], dtype=np.float32)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-6:
        return default_normal
    normal = normal / norm
    if float(np.dot(normal, default_normal)) < 0.0:
        normal = -normal
    return normal.astype(np.float32)


def smooth_normal(prev: np.ndarray, new: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """EMA-blend two unit normals and renormalize.

    Used to refresh the table normal per frame (eye-in-hand) without jitter: `alpha`
    is how much of the freshly estimated normal to take each update. Both inputs are
    assumed to already point to the same hemisphere (estimate_table_normal enforces it).
    """
    prev = np.asarray(prev, dtype=np.float32)
    new = np.asarray(new, dtype=np.float32)
    blended = (1.0 - alpha) * prev + alpha * new
    norm = float(np.linalg.norm(blended))
    if norm < 1e-6:
        return new
    return (blended / norm).astype(np.float32)


def frame_output_record(
    frame_index: int,
    fps_value: float,
    infer_ms: float,
    geom_ms: float,
    scene_points: np.ndarray,
    table_normal: np.ndarray,
    detections_3d: list[Detection3D],
) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "fps": round(float(fps_value), 4),
        "infer_ms": round(float(infer_ms), 4),
        "geom_ms": round(float(geom_ms), 4),
        "scene_point_count": int(len(scene_points)),
        "table_normal_xyz": [round(float(v), 6) for v in table_normal.tolist()],
        "detections": [_detection_record(det) for det in detections_3d if det.center_xyz is not None and det.in_workspace],
    }


def _round_list(values, ndigits: int):
    return None if values is None else [round(float(v), ndigits) for v in values]


def _detection_record(det: Any) -> dict[str, Any]:
    """Serialize one detection. Raw per-frame fields always; temporal fields
    (filtered pose, dispersion, identity, stable) only when present (tracker output)."""
    record = {
        "class_name": det.class_name,
        "confidence": round(float(det.confidence), 6),
        "center_camera_xyz_m": _round_list(det.center_xyz, 6),
        "extent_xyz_m": _round_list(det.extent_xyz, 6),
        "yaw_rad": None if det.yaw_rad is None else round(float(det.yaw_rad), 6),
        "yaw_deg": None if det.yaw_deg is None else round(float(det.yaw_deg), 4),
        "yaw_reliable": bool(getattr(det, "yaw_reliable", False)),
        "point_count": int(det.point_count),
    }
    if getattr(det, "stable", None) is not None:  # tracker output (TrackedDetection)
        record.update(
            {
                "track_id": int(det.track_id),
                "hits": int(det.hits),
                "confidence_mean": round(float(det.confidence_mean), 6),
                "center_filtered_camera_xyz_m": _round_list(det.center_filtered_xyz, 6),
                "yaw_filtered_deg": round(float(det.yaw_filtered_deg), 4),
                "position_std_m": round(float(det.position_std_m), 6),
                "yaw_std_deg": round(float(det.yaw_std_deg), 4),
                "stable": bool(det.stable),
            }
        )
    return record


def filtered_box_corners(detection: Any, table_normal: np.ndarray) -> np.ndarray | None:
    """Corners from the temporally filtered pose when available (tracker output),
    else the raw per-frame corners.

    Recomputes the 8 OBB corners from the filtered center/extent/yaw on the table
    frame, so the viewer can draw a box that matches the smoothed numbers shown in
    the label/JSON instead of the jittery instantaneous box.
    """
    yaw_filtered = getattr(detection, "yaw_filtered_rad", None)
    center_filtered = getattr(detection, "center_filtered_xyz", None)
    extent_filtered = getattr(detection, "extent_filtered_xyz", None)
    if yaw_filtered is None or center_filtered is None or extent_filtered is None:
        return detection.box_corners_xyz
    return obb_corners_from_pose(center_filtered, extent_filtered, yaw_filtered, table_normal)
