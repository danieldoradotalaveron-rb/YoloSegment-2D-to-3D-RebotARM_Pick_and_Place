"""Initialize labeled 3D Gaussians from RGBD captures (lift in RAM, no seeds on disk).

Usage:
  python DS/prelabel/init_labeled_gaussians.py --inpaint-depth
  python DS/prelabel/init_labeled_gaussians.py --capture capture_000004 --inpaint-depth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from labeled_gaussians import (
    LabeledGaussians,
    merge_instance_points,
    points_to_gaussians,
    save_gaussians,
)
from lift_rgbd import (
    DEFAULT_CLASSES_YAML,
    DEFAULT_RGBD_ROOT,
    lift_capture,
    load_class_map,
    summarize_rgbd_captures,
)

DS_DIR = SCRIPT_DIR.parent
DEFAULT_SYNTH_ROOT = DS_DIR / "dataset_prelabel" / "synth_render_point"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lift RGBD + init labeled Gaussians (in RAM -> npz build). "
        "Default: all labeled captures under --rgbd-root."
    )
    parser.add_argument("--rgbd-root", type=Path, default=DEFAULT_RGBD_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_SYNTH_ROOT)
    parser.add_argument(
        "--capture",
        default="",
        help="Single capture id (default: all labeled captures under rgbd-root).",
    )
    parser.add_argument("--classes-yaml", type=Path, default=DEFAULT_CLASSES_YAML)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=2.0)
    parser.add_argument("--inpaint-depth", action="store_true")
    parser.add_argument("--inpaint-radius", type=int, default=5)
    parser.add_argument("--scale-m", type=float, default=0.003, help="Initial Gaussian axis scale (m).")
    parser.add_argument("--opacity", type=float, default=0.85, help="Initial opacity 0..1.")
    return parser.parse_args()


def init_capture_gaussians(
    capture_dir: Path,
    class_map: dict[str, int],
    *,
    min_depth_m: float,
    max_depth_m: float,
    inpaint_depth: bool,
    inpaint_radius: int,
    scale_m: float,
    opacity: float,
) -> tuple[LabeledGaussians, dict]:
    cap = lift_capture(
        capture_dir,
        class_map=class_map,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        inpaint_depth=inpaint_depth,
        inpaint_radius=inpaint_radius,
    )
    chunks: list[tuple] = []
    for inst in cap.instance_chunks:
        rgb01 = inst.colors_rgb.astype("float32") / 255.0
        chunks.append((inst.points_xyz, rgb01, inst.class_id, inst.instance_id))

    xyz, rgb, class_id, instance_id = merge_instance_points(chunks)
    gaussians = points_to_gaussians(
        xyz,
        rgb,
        class_id,
        instance_id,
        capture_id=cap.capture_id,
        scale_m=scale_m,
        opacity=opacity,
    )
    meta = {
        "capture_id": cap.capture_id,
        "capture_dir": str(capture_dir),
        "gaussian_count": gaussians.count,
        "instances": [
            {
                "instance_id": i.instance_id,
                "class_name": i.class_name,
                "class_id": i.class_id,
                "point_count": i.point_count,
            }
            for i in cap.instances
        ],
        "intrinsics": cap.intrinsics,
    }
    return gaussians, meta


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

    out_root = args.output_root.resolve()
    ok: list[tuple[str, int]] = []
    skipped: list[tuple[str, str]] = []

    print(f"Initializing gaussians for {len(labeled_dirs)} labeled capture(s) under {rgbd_root}")
    for capture_dir in labeled_dirs:
        try:
            gaussians, meta = init_capture_gaussians(
                capture_dir,
                class_map,
                min_depth_m=args.min_depth,
                max_depth_m=args.max_depth,
                inpaint_depth=args.inpaint_depth,
                inpaint_radius=args.inpaint_radius,
                scale_m=args.scale_m,
                opacity=args.opacity,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"SKIP {capture_dir.name}: {exc}")
            skipped.append((capture_dir.name, str(exc)))
            continue

        out_dir = out_root / capture_dir.name
        out_npz = out_dir / "gaussians.npz"
        save_gaussians(out_npz, gaussians, meta=meta)
        print(f"  {capture_dir.name}: {gaussians.count} gaussians -> {out_npz}")
        ok.append((capture_dir.name, gaussians.count))

    print(f"\n=== Init gaussians summary ===")
    print(f"OK: {len(ok)}/{len(labeled_dirs)} capture(s)")
    for name, count in ok:
        print(f"  {name}: {count} gaussians")
    if skipped:
        print(f"Skipped: {len(skipped)}")
        for name, reason in skipped:
            print(f"  {name}: {reason}")
    if not ok:
        raise SystemExit("No gaussians initialized.")


if __name__ == "__main__":
    main()
