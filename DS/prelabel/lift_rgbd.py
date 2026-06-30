"""Lift LabelMe polygons on RGBD captures to 3D colored points (Phase 1B test).

Reads each folder under dataset_capture/rgbd/capture_*/
  rgb.png, depth.npy, intrinsics.yaml, rgb.json (LabelMe)

Usage:
  python DS/prelabel/lift_rgbd.py
  python DS/prelabel/lift_rgbd.py --capture capture_000001
  python DS/prelabel/lift_rgbd.py --save-ply   # write lift_<capture>.ply per scene
  python DS/prelabel/lift_rgbd.py --inpaint-depth   # fill small depth holes inside masks
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DS_DIR = SCRIPT_DIR.parent
REPO_ROOT = DS_DIR.parent
GEOMETRY_SRC = REPO_ROOT / "TabletopSeg3D" / "3DDetection" / "src"
if str(GEOMETRY_SRC) not in sys.path:
    sys.path.insert(0, str(GEOMETRY_SRC))

from geometry.pointcloud import project_mask_to_colored_points, _valid_mask  # noqa: E402

DEFAULT_RGBD_ROOT = DS_DIR / "dataset_capture" / "rgbd"
DEFAULT_CLASSES_YAML = DS_DIR / "yolo_classes.yaml"
# Build output (regenerable, .gitignored). Raw captures under rgbd/ are never written to.
DEFAULT_PLY_ROOT = DS_DIR / "dataset_prelabel" / "synth_render_point"
LABEL_JSON_NAMES = ("rgb.json", "labelme.json")
RGB_NAME = "rgb.png"
# Labels starting with this prefix are occluders (e.g. the robot gripper), not
# object classes: they are used only to mask composited views, never lifted to 3D
# nor added to yolo_classes.yaml / the trained model.
OCCLUDER_PREFIX = "_"


@dataclass
class InstanceLift:
    capture_id: str
    instance_id: int
    class_name: str
    class_id: int
    point_count: int
    mask_pixels: int
    depth_ok_pct: float
    z_min: float
    z_max: float
    z_median: float


@dataclass
class InstancePoints:
    instance_id: int
    class_name: str
    class_id: int
    points_xyz: np.ndarray
    colors_rgb: np.ndarray


@dataclass
class CaptureLift:
    capture_id: str
    capture_dir: Path
    intrinsics: dict
    instances: list[InstanceLift] = field(default_factory=list)
    instance_chunks: list[InstancePoints] = field(default_factory=list)
    all_points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    all_colors_rgb: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.uint8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lift RGBD LabelMe masks to 3D points.")
    parser.add_argument(
        "--rgbd-root",
        type=Path,
        default=DEFAULT_RGBD_ROOT,
        help="Root folder with capture_*/ subdirs.",
    )
    parser.add_argument(
        "--capture",
        default="",
        help="Process only this capture id (e.g. capture_000001). Default: all labeled captures.",
    )
    parser.add_argument("--min-depth", type=float, default=0.05, help="Min valid depth (m). Offline band, not tabletop.")
    parser.add_argument("--max-depth", type=float, default=2.0, help="Max valid depth (m).")
    parser.add_argument(
        "--classes-yaml",
        type=Path,
        default=DEFAULT_CLASSES_YAML,
        help="Map label strings to class_id (yolo_classes.yaml).",
    )
    parser.add_argument(
        "--save-ply",
        action="store_true",
        help="Write lift_<capture_id>.ply (Open3D) next to each capture for visual inspection.",
    )
    parser.add_argument(
        "--inpaint-depth",
        action="store_true",
        help="Fill depth==0 holes inside label masks before lifting (OpenCV inpaint).",
    )
    parser.add_argument(
        "--inpaint-radius",
        type=int,
        default=5,
        help="Inpaint radius in pixels (default: 5).",
    )
    return parser.parse_args()


def load_class_map(path: Path) -> dict[str, int]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    classes = data.get("classes") or []
    return {name: idx for idx, name in enumerate(classes)}


def find_label_json(capture_dir: Path) -> Path | None:
    for name in LABEL_JSON_NAMES:
        candidate = capture_dir / name
        if candidate.is_file():
            return candidate
    return None


def polygon_to_mask(points: list, height: int, width: int) -> np.ndarray:
    contour = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    mask = np.zeros((height, width), dtype=bool)
    cv2.fillPoly(mask.view(np.uint8), [contour.astype(np.int32)], 1)
    return mask.astype(bool)


def load_intrinsics(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def inpaint_depth_in_mask(
    depth_m: np.ndarray,
    mask: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, int]:
    """Fill invalid depth (<=0) inside mask using OpenCV inpaint on a normalized depth image."""
    depth = depth_m.astype(np.float32, copy=True)
    invalid = mask & ((depth <= 0) | ~np.isfinite(depth))
    n_holes = int(invalid.sum())
    if n_holes == 0:
        return depth, 0

    valid_in_mask = mask & (depth > 0) & np.isfinite(depth)
    if not np.any(valid_in_mask):
        return depth_m, 0

    vals = depth[valid_in_mask]
    lo = float(vals.min())
    hi = float(vals.max())
    span = max(hi - lo, 1e-4)

    norm = np.zeros(depth.shape, dtype=np.float32)
    norm[valid_in_mask] = (depth[valid_in_mask] - lo) / span * 255.0
    img_u8 = np.clip(norm, 0, 255).astype(np.uint8)
    hole_u8 = invalid.astype(np.uint8) * 255
    filled_u8 = cv2.inpaint(img_u8, hole_u8, max(1, radius), cv2.INPAINT_NS)

    depth[invalid] = lo + (filled_u8[invalid].astype(np.float32) / 255.0) * span
    return depth, n_holes


def collect_label_shapes(label_data: dict) -> list[dict]:
    shapes = []
    for shape in label_data.get("shapes", []):
        if shape.get("shape_type") != "polygon":
            continue
        points = shape.get("points") or []
        if len(points) < 3:
            continue
        label = (shape.get("label") or "").strip()
        if label:
            shapes.append(shape)
    return shapes


def lift_capture(
    capture_dir: Path,
    class_map: dict[str, int],
    min_depth_m: float,
    max_depth_m: float,
    *,
    inpaint_depth: bool = False,
    inpaint_radius: int = 5,
) -> CaptureLift:
    capture_id = capture_dir.name
    rgb_path = capture_dir / RGB_NAME
    depth_path = capture_dir / "depth.npy"
    intrinsics_path = capture_dir / "intrinsics.yaml"
    label_path = find_label_json(capture_dir)

    for required in (rgb_path, depth_path, intrinsics_path, label_path):
        if required is None or not required.is_file():
            missing = required or "label json"
            raise FileNotFoundError(f"{capture_id}: missing {missing}")

    rgb = cv2.imread(str(rgb_path))
    if rgb is None:
        raise RuntimeError(f"{capture_id}: cannot read {rgb_path}")

    depth_m = np.load(depth_path).astype(np.float32)
    intrinsics = load_intrinsics(intrinsics_path)
    label_data = json.loads(label_path.read_text(encoding="utf-8"))

    height, width = rgb.shape[:2]
    if depth_m.shape[:2] != (height, width):
        raise RuntimeError(
            f"{capture_id}: depth {depth_m.shape[:2]} != rgb {(height, width)}"
        )

    shapes = collect_label_shapes(label_data)
    if inpaint_depth and shapes:
        union_mask = np.zeros((height, width), dtype=bool)
        for shape in shapes:
            union_mask |= polygon_to_mask(shape["points"], height, width)
        depth_m, n_filled = inpaint_depth_in_mask(depth_m, union_mask, inpaint_radius)
        if n_filled:
            print(f"  {capture_id}: inpaint filled {n_filled} depth hole pixel(s) inside mask(s)")

    result = CaptureLift(
        capture_id=capture_id,
        capture_dir=capture_dir,
        intrinsics=intrinsics,
    )
    point_chunks: list[np.ndarray] = []
    color_chunks: list[np.ndarray] = []

    instance_id = 0
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        if label.startswith(OCCLUDER_PREFIX):
            continue  # occluder mask (composite-only), never lifted to 3D
        if label not in class_map:
            print(f"  WARN {capture_id}: unknown label '{label}' (add to yolo_classes.yaml)")
            continue

        instance_id += 1
        mask = polygon_to_mask(points, height, width)
        mask_pixels = int(mask.sum())
        depth_ok = int(_valid_mask(mask, depth_m, min_depth_m, max_depth_m).sum())
        depth_ok_pct = 100.0 * depth_ok / mask_pixels if mask_pixels else 0.0
        xyz, colors_bgr, _ = project_mask_to_colored_points(
            mask=mask,
            depth_m=depth_m,
            rgb=rgb,
            intrinsics=intrinsics,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
        )

        z_vals = xyz[:, 2] if len(xyz) else np.array([], dtype=np.float32)
        inst = InstanceLift(
            capture_id=capture_id,
            instance_id=instance_id,
            class_name=label,
            class_id=class_map[label],
            point_count=int(len(xyz)),
            mask_pixels=mask_pixels,
            depth_ok_pct=depth_ok_pct,
            z_min=float(z_vals.min()) if len(z_vals) else 0.0,
            z_max=float(z_vals.max()) if len(z_vals) else 0.0,
            z_median=float(np.median(z_vals)) if len(z_vals) else 0.0,
        )
        result.instances.append(inst)

        colors_rgb = colors_bgr[:, ::-1] if len(xyz) else np.empty((0, 3), dtype=np.uint8)
        result.instance_chunks.append(
            InstancePoints(
                instance_id=instance_id,
                class_name=label,
                class_id=class_map[label],
                points_xyz=xyz,
                colors_rgb=colors_rgb,
            )
        )

        if len(xyz):
            point_chunks.append(xyz)
            color_chunks.append(colors_rgb)

    if point_chunks:
        result.all_points = np.vstack(point_chunks)
        result.all_colors_rgb = np.vstack(color_chunks)

    return result


def save_ply(capture: CaptureLift, ply_root: Path = DEFAULT_PLY_ROOT) -> Path | None:
    if len(capture.all_points) == 0:
        return None

    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit("open3d is required for --save-ply") from exc

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(capture.all_points.astype(np.float64))
    cloud.colors = o3d.utility.Vector3dVector(
        capture.all_colors_rgb.astype(np.float64) / 255.0
    )
    out_dir = ply_root / capture.capture_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"lift_{capture.capture_id}.ply"
    o3d.io.write_point_cloud(str(out), cloud, write_ascii=False)
    return out


def iter_capture_dirs(rgbd_root: Path, capture_filter: str) -> list[Path]:
    if not rgbd_root.is_dir():
        raise SystemExit(f"Not found: {rgbd_root}")

    dirs = sorted(p for p in rgbd_root.iterdir() if p.is_dir() and p.name.startswith("capture_"))
    if capture_filter:
        dirs = [p for p in dirs if p.name == capture_filter]
        if not dirs:
            raise SystemExit(f"Capture not found: {capture_filter} under {rgbd_root}")
    return dirs


def list_labeled_capture_dirs(rgbd_root: Path, capture_filter: str = "") -> list[Path]:
    """All capture_* folders under rgbd_root that have LabelMe JSON (rgb.json / labelme.json)."""
    return [d for d in iter_capture_dirs(rgbd_root, capture_filter) if find_label_json(d)]


def summarize_rgbd_captures(
    rgbd_root: Path, capture_filter: str = ""
) -> tuple[list[Path], list[Path]]:
    """Return (labeled, unlabeled) capture dirs under rgbd_root."""
    all_dirs = iter_capture_dirs(rgbd_root, capture_filter)
    labeled = [d for d in all_dirs if find_label_json(d)]
    labeled_set = set(labeled)
    unlabeled = [d for d in all_dirs if d not in labeled_set]
    return labeled, unlabeled


def print_report(captures: list[CaptureLift]) -> None:
    total_instances = 0
    total_points = 0

    print("\n=== RGBD lift report ===")
    for cap in captures:
        cap_points = int(len(cap.all_points))
        total_points += cap_points
        total_instances += len(cap.instances)
        print(f"\n{cap.capture_id}  instances={len(cap.instances)}  points={cap_points}")
        for inst in cap.instances:
            print(
                f"  #{inst.instance_id:02d} {inst.class_name:<8} "
                f"id={inst.class_id}  N={inst.point_count:5d}/{inst.mask_pixels} "
                f"({inst.depth_ok_pct:4.1f}% depth)  "
                f"z=[{inst.z_min:.3f}, {inst.z_median:.3f}, {inst.z_max:.3f}] m"
            )
        if any(inst.depth_ok_pct < 95.0 for inst in cap.instances):
            print("  ^ depth holes in mask (common on metal/glossy parts — not a lifting bug)")

    print(f"\nTotal: {len(captures)} capture(s), {total_instances} instance(s), {total_points} point(s)")
    if total_points == 0:
        print("No points lifted — check labels, depth holes, or depth band (--min-depth / --max-depth).")


def main() -> None:
    args = parse_args()
    class_map = load_class_map(args.classes_yaml.resolve())
    rgbd_root = args.rgbd_root.resolve()
    labeled_dirs, unlabeled_dirs = summarize_rgbd_captures(rgbd_root, args.capture.strip())
    if unlabeled_dirs and not args.capture.strip():
        names = ", ".join(d.name for d in unlabeled_dirs)
        print(f"Skipping {len(unlabeled_dirs)} capture(s) without labels: {names}")
    if not labeled_dirs:
        raise SystemExit(f"No labeled captures (rgb.json) under {rgbd_root}")

    captures: list[CaptureLift] = []
    for capture_dir in labeled_dirs:
        try:
            cap = lift_capture(
                capture_dir,
                class_map=class_map,
                min_depth_m=args.min_depth,
                max_depth_m=args.max_depth,
                inpaint_depth=args.inpaint_depth,
                inpaint_radius=args.inpaint_radius,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"SKIP {capture_dir.name}: {exc}")
            continue
        captures.append(cap)

    print_report(captures)

    if args.save_ply:
        for cap in captures:
            out = save_ply(cap)
            if out:
                print(f"saved: {out}")


if __name__ == "__main__":
    main()
