"""Labeled 3D Gaussian initialization from lifted RGBD points (Phase 1B)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class LabeledGaussians:
    """One Gaussian per lifted point; class/instance ids are constant attributes."""

    xyz: np.ndarray  # (N, 3) float32, camera frame, meters
    rgb: np.ndarray  # (N, 3) float32, 0..1
    scales_log: np.ndarray  # (N, 3) float32, log axis scales (meters)
    quats_wxyz: np.ndarray  # (N, 4) float32, identity by default
    opacity: np.ndarray  # (N,) float32, 0..1
    class_id: np.ndarray  # (N,) int32
    instance_id: np.ndarray  # (N,) int32
    capture_id: str

    @property
    def count(self) -> int:
        return int(len(self.xyz))


def points_to_gaussians(
    xyz: np.ndarray,
    rgb01: np.ndarray,
    class_id: np.ndarray,
    instance_id: np.ndarray,
    capture_id: str,
    *,
    scale_m: float = 0.003,
    opacity: float = 0.85,
) -> LabeledGaussians:
    """Convert lifted colored points into initial Gaussian parameters."""
    n = len(xyz)
    if n == 0:
        empty_f = np.empty((0,), dtype=np.float32)
        return LabeledGaussians(
            xyz=np.empty((0, 3), dtype=np.float32),
            rgb=np.empty((0, 3), dtype=np.float32),
            scales_log=np.empty((0, 3), dtype=np.float32),
            quats_wxyz=np.empty((0, 4), dtype=np.float32),
            opacity=empty_f,
            class_id=np.empty((0,), dtype=np.int32),
            instance_id=np.empty((0,), dtype=np.int32),
            capture_id=capture_id,
        )

    log_scale = float(np.log(max(scale_m, 1e-6)))
    scales_log = np.full((n, 3), log_scale, dtype=np.float32)
    quats = np.zeros((n, 4), dtype=np.float32)
    quats[:, 0] = 1.0  # wxyz identity
    opacities = np.full(n, opacity, dtype=np.float32)

    return LabeledGaussians(
        xyz=xyz.astype(np.float32, copy=False),
        rgb=rgb01.astype(np.float32, copy=False),
        scales_log=scales_log,
        quats_wxyz=quats,
        opacity=opacities,
        class_id=class_id.astype(np.int32, copy=False),
        instance_id=instance_id.astype(np.int32, copy=False),
        capture_id=capture_id,
    )


def merge_instance_points(
    instance_chunks: list[tuple[np.ndarray, np.ndarray, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Merge per-instance (xyz, rgb01, class_id, instance_id) lists."""
    if not instance_chunks:
        z = np.empty((0, 3), dtype=np.float32)
        c = np.empty((0, 3), dtype=np.float32)
        ci = np.empty((0,), dtype=np.int32)
        ii = np.empty((0,), dtype=np.int32)
        return z, c, ci, ii

    xyz_parts: list[np.ndarray] = []
    rgb_parts: list[np.ndarray] = []
    class_parts: list[np.ndarray] = []
    inst_parts: list[np.ndarray] = []
    for xyz, rgb, class_id, instance_id in instance_chunks:
        n = len(xyz)
        if n == 0:
            continue
        xyz_parts.append(xyz)
        rgb_parts.append(rgb)
        class_parts.append(np.full(n, class_id, dtype=np.int32))
        inst_parts.append(np.full(n, instance_id, dtype=np.int32))

    return (
        np.vstack(xyz_parts),
        np.vstack(rgb_parts),
        np.concatenate(class_parts),
        np.concatenate(inst_parts),
    )


def save_gaussians(path: Path, gaussians: LabeledGaussians, meta: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        xyz=gaussians.xyz,
        rgb=gaussians.rgb,
        scales_log=gaussians.scales_log,
        quats_wxyz=gaussians.quats_wxyz,
        opacity=gaussians.opacity,
        class_id=gaussians.class_id,
        instance_id=gaussians.instance_id,
        capture_id=np.array(gaussians.capture_id),
    )
    if meta is not None:
        meta_path = path.with_suffix(".meta.yaml")
        meta_path.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")


def load_gaussians(path: Path) -> LabeledGaussians:
    data = np.load(path)
    capture_raw = data["capture_id"]
    capture_id = capture_raw.item() if hasattr(capture_raw, "item") else str(capture_raw)
    return LabeledGaussians(
        xyz=data["xyz"].astype(np.float32),
        rgb=data["rgb"].astype(np.float32),
        scales_log=data["scales_log"].astype(np.float32),
        quats_wxyz=data["quats_wxyz"].astype(np.float32),
        opacity=data["opacity"].astype(np.float32),
        class_id=data["class_id"].astype(np.int32),
        instance_id=data["instance_id"].astype(np.int32),
        capture_id=str(capture_id),
    )


def load_gaussians_meta(path: Path) -> dict[str, Any]:
    meta_path = path.with_suffix(".meta.yaml")
    if not meta_path.is_file():
        return {}
    return yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
