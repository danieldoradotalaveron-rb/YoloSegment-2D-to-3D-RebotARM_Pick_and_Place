"""
Checks the hand-eye calibration by verifying that a fixed object lands on the same
base-frame position when seen from several arm poses.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

HANDEYE_DIR = Path(__file__).resolve().parents[0]
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "TabletopSeg3D/3DDetection/src"))
sys.path.insert(0, str(HANDEYE_DIR))

from camera.realsense_capture import (  # noqa: E402
    build_runtime,
    enumerate_devices,
    get_aligned_frame_bundle,
    stop_runtimes,
)
from robot.extrinsics import cam_point_to_base  # noqa: E402
from pipeline import (  # noqa: E402
    build_detection_3d,
    build_scene_point_cloud,
    estimate_table_normal,
    load_model,
    parse_class_set,
    run_inference,
    smooth_normal,
)
from robot.robot_pose import RobotPoseReader  # noqa: E402
from tracking import DetectionTracker, TrackerConfig  # noqa: E402


def find_latest_ee_T_cam() -> str:
    candidates = sorted(
        (HANDEYE_DIR / "calibration_d405" / "captures").glob("session_*/ee_T_cam.json"),
        key=lambda p: p.stat().st_mtime,
    )
    return str(candidates[-1]) if candidates else ""


def find_latest_best_pt() -> str:
    """Latest trained YOLO-seg weights, mirroring how sense.sh picks the model."""
    candidates = sorted(
        (REPO_ROOT / "runs" / "segment").glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    return str(candidates[-1]) if candidates else "yolo26m-seg.pt"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ee-t-cam", default=find_latest_ee_T_cam(), help="Path to ee_T_cam.json (default: latest session).")
    ap.add_argument("--class", dest="target_class", default="", help="Only validate this class (e.g. cork).")
    ap.add_argument("--base-frame", default="base_link")
    ap.add_argument("--ee-frame", default="end_link")
    # Camera / inference (mirror the runtime defaults).
    ap.add_argument("--serial", default="419522072950")
    ap.add_argument("--model", default=find_latest_best_pt())
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=448)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--dedup-iou", type=float, default=0.7)
    ap.add_argument("--max-det", type=int, default=10)
    ap.add_argument("--min-depth", type=float, default=0.10)
    ap.add_argument("--max-depth", type=float, default=1.50)
    ap.add_argument("--min-points", type=int, default=500)
    ap.add_argument("--non-symmetric-classes", default="")
    ap.add_argument("--point-stride", type=int, default=2)
    ap.add_argument("--scene-max-points", type=int, default=80000)
    ap.add_argument("--warmup-frames", type=int, default=5)
    ap.add_argument("--table-normal-every", type=int, default=1)
    ap.add_argument("--tf-timeout", type=float, default=0.5)
    args = ap.parse_args()
    args.workspace_bounds = None
    args.non_symmetric_set = parse_class_set(args.non_symmetric_classes)
    return args


def pick_target(tracked: list, target_class: str):
    """Among stable, in-workspace tracks, return the best target (class filter +
    highest mean confidence), or None."""
    candidates = [
        d for d in tracked
        if getattr(d, "stable", False)
        and getattr(d, "in_workspace", True)
        and (not target_class or d.class_name == target_class)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.confidence_mean)


def report_spread(samples: list[dict]) -> None:
    if len(samples) < 2:
        print(f"\n[result] only {len(samples)} sample(s); need >= 2 to measure spread.")
        return
    arr = np.array([s["base_xyz"] for s in samples], dtype=np.float64)
    mean = arr.mean(axis=0)
    per_axis_std = arr.std(axis=0)
    norm_std = float(np.linalg.norm(per_axis_std))
    max_pair = max(float(np.linalg.norm(a - b)) for a, b in combinations(arr, 2))
    print("\n==== base-frame consistency ====")
    print(f"samples           : {len(arr)}")
    print(f"mean base xyz (m) : {np.round(mean, 4).tolist()}")
    print(f"std per axis (mm) : {np.round(per_axis_std * 1000, 2).tolist()}")
    print(f"std norm (mm)     : {norm_std * 1000:.2f}")
    print(f"max pairwise (mm) : {max_pair * 1000:.2f}")
    verdict = "GOOD (<= 7 mm)" if norm_std * 1000 <= 7.0 else "HIGH (recapture hand-eye closer/frontal)"
    print(f"verdict           : {verdict}")


def main() -> int:
    args = parse_args()
    if not args.ee_t_cam or not Path(args.ee_t_cam).exists():
        print(f"[error] ee_T_cam.json not found: {args.ee_t_cam!r}. Pass --ee-t-cam.")
        return 1
    ee_T_cam = np.array(json.loads(Path(args.ee_t_cam).read_text())["ee_T_cam"], dtype=np.float64)
    print(f"[extrinsics] loaded ee_T_cam from {args.ee_t_cam}")

    devices = enumerate_devices()
    if not devices:
        print("[error] no RealSense device found")
        return 1
    device = next((d for d in devices if d.serial_number == args.serial), devices[0])
    print(f"[realsense] using {device.name} ({device.serial_number})")
    print(f"[model] {args.model}")
    model = load_model(args.model)
    runtime = build_runtime(device, args.width, args.height, args.fps)
    pose_reader = RobotPoseReader(args.base_frame, args.ee_frame, node_name="handeye_validate")
    tracker = DetectionTracker(TrackerConfig())

    import open3d as o3d

    table_normal = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    for _ in range(args.warmup_frames):
        get_aligned_frame_bundle(runtime, args.min_depth, args.max_depth)

    samples: list[dict] = []
    window = "Fase A validation (c=capture  q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    status = "place object, hold arm still, wait for STABLE, press c"
    frame_counter = 0
    try:
        while True:
            bundle = get_aligned_frame_bundle(runtime, args.min_depth, args.max_depth)
            depth_m = bundle["depth"].astype(np.float32) * float(bundle["depth_scale"])
            scene_points, _ = build_scene_point_cloud(
                color_image=bundle["color"], depth_m=depth_m,
                intrinsics=bundle["aligned_intrinsics"], args=args,
            )
            if args.table_normal_every > 0 and frame_counter % args.table_normal_every == 0:
                table_normal = smooth_normal(table_normal, estimate_table_normal(scene_points, o3d))
            detections = run_inference(model, bundle["color"], args)
            detections_3d = [
                build_detection_3d(det, depth_m, bundle["aligned_intrinsics"], table_normal, args)
                for det in detections
            ]
            tracked = tracker.update(detections_3d)
            target = pick_target(tracked, args.target_class)
            frame_counter += 1

            preview = bundle["color"].copy()
            base_xyz_live = None
            if target is not None:
                cam_xyz = np.asarray(target.center_filtered_xyz, dtype=np.float64)
                base_T_ee = pose_reader.lookup(args.tf_timeout)
                if base_T_ee is not None:
                    base_xyz_live = cam_point_to_base(cam_xyz, base_T_ee, ee_T_cam)
                dist_cm = float(cam_xyz[2]) * 100.0
                offaxis_cm = float(np.hypot(cam_xyz[0], cam_xyz[1])) * 100.0
                # Good view for validation: ~25-35 cm away and well centered (< 6 cm off axis).
                good = (25.0 <= dist_cm <= 35.0) and (offaxis_cm < 6.0)
                geo_color = (0, 200, 0) if good else (0, 165, 255)
                line = f"target={target.class_name} STABLE cam={np.round(cam_xyz, 3).tolist()}"
                cv2.putText(preview, line, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
                cv2.putText(preview, f"dist={dist_cm:.1f} cm  offaxis={offaxis_cm:.1f} cm  {'OK view' if good else 'move to ~30cm/center'}",
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, geo_color, 2)
                if base_xyz_live is not None:
                    cv2.putText(preview, f"base={np.round(base_xyz_live, 3).tolist()}",
                                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
                else:
                    cv2.putText(preview, "base=(no TF)", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.putText(preview, "no stable target", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(preview, f"{status}  [{len(samples)} captured]",
                        (10, preview.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("c"), ord("C"), ord(" ")):
                if target is None:
                    status = "REJECTED: no stable target"
                    continue
                if base_xyz_live is None:
                    status = "REJECTED: no TF base_T_ee"
                    continue
                samples.append({
                    "class_name": target.class_name,
                    "cam_xyz": np.asarray(target.center_filtered_xyz, dtype=float).tolist(),
                    "base_xyz": [float(v) for v in base_xyz_live],
                })
                status = f"CAPTURED #{len(samples)-1}: base={np.round(base_xyz_live, 3).tolist()}"
                print(f"[capture] {status}")
    finally:
        cv2.destroyAllWindows()
        stop_runtimes([runtime])
        pose_reader.shutdown()

    report_spread(samples)
    if samples:
        out = HANDEYE_DIR / "validation_base_frame.json"
        out.write_text(json.dumps({"ee_t_cam": args.ee_t_cam, "samples": samples}, indent=2))
        print(f"[written] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
