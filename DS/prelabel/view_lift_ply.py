"""View a lift_*.ply with the same orientation as the RealSense capture (rgb.png).

Open3D defaults to Y-up; RealSense camera frame is X-right, Y-down, Z-forward.
This viewer sets front/up to match the capture view so objects are not "rotated".

Usage:
  python DS/prelabel/view_lift_ply.py --capture capture_000006
  python DS/prelabel/view_lift_ply.py --ply path/to/lift_capture_000001.ply
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DS_DIR = SCRIPT_DIR.parent
DEFAULT_RGBD_ROOT = DS_DIR / "dataset_capture" / "rgbd"
DEFAULT_PLY_ROOT = DS_DIR / "dataset_prelabel" / "synth_render_point"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View lift PLY aligned to capture camera.")
    parser.add_argument("--capture", default="", help="Capture id, e.g. capture_000006")
    parser.add_argument("--ply", type=Path, default=None, help="Direct path to .ply")
    parser.add_argument("--rgbd-root", type=Path, default=DEFAULT_RGBD_ROOT)
    parser.add_argument("--ply-root", type=Path, default=DEFAULT_PLY_ROOT)
    parser.add_argument("--point-size", type=float, default=3.0)
    return parser.parse_args()


def resolve_ply(args: argparse.Namespace) -> Path:
    if args.ply is not None:
        ply = args.ply.resolve()
        if not ply.is_file():
            raise SystemExit(f"Not found: {ply}")
        return ply
    if not args.capture:
        raise SystemExit("Pass --capture capture_XXXXXX or --ply path/to/file.ply")
    name = f"lift_{args.capture}.ply"
    # Build location first; fall back to legacy in-capture location.
    candidates = [
        (args.ply_root / args.capture / name).resolve(),
        (args.rgbd_root / args.capture / name).resolve(),
    ]
    for ply in candidates:
        if ply.is_file():
            return ply
    raise SystemExit(
        f"Not found: {candidates[0]}  (run: just lift-rgbd --capture {args.capture} --save-ply)"
    )


def scene_center(ply_path: Path) -> np.ndarray:
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(ply_path))
    if len(cloud.points) == 0:
        return np.array([0.0, 0.0, 0.4], dtype=np.float64)
    pts = np.asarray(cloud.points)
    return pts.mean(axis=0)


def median_depth(ply_path: Path) -> float:
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(ply_path))
    pts = np.asarray(cloud.points)
    if len(pts) == 0:
        return 0.4
    return float(np.median(pts[:, 2]))


def main() -> None:
    args = parse_args()
    ply_path = resolve_ply(args)
    center = scene_center(ply_path)
    z_med = median_depth(ply_path)

    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(ply_path))
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    print(f"Viewing: {ply_path}")
    print("Camera frame: X=red (right), Y=green (down in image), Z=blue (forward / depth)")
    print("Controls: drag=orbit, scroll=zoom, Shift+drag=pan, Q=quit")
    print("Tip: view should match rgb.png orientation (not upside-down).")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"lift view: {ply_path.name}", width=960, height=720)
    vis.add_geometry(cloud)
    vis.add_geometry(frame)
    render = vis.get_render_option()
    render.point_size = float(args.point_size)
    render.background_color = np.array([0.95, 0.95, 0.95])

    ctrl = vis.get_view_control()
    # RealSense optical frame: look along +Z, image Y points down -> up vector is -Y.
    ctrl.set_front([0.0, 0.0, 1.0])
    ctrl.set_up([0.0, -1.0, 0.0])
    ctrl.set_lookat(center.tolist())
    ctrl.set_zoom(0.55 if z_med < 0.6 else 0.45)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
