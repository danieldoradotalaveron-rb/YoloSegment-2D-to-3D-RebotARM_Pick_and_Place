#!/usr/bin/env python3
"""Perception ROS2 node: publish stable 3D detections in the robot base frame.

Headless sibling of realtime_open3d_scene.py. Reuses the same pipeline (YOLO-seg
-> 3D -> per-frame table normal -> tracker), and for every STABLE, in-workspace
detection it transforms the filtered camera-frame center to base_link
(base_T_ee from TF x ee_T_cam from hand-eye) and publishes a
vision_msgs/Detection3DArray.

This is a plain rclpy participant (NOT a colcon package): it runs in the repo
.venv so it has open3d/torch/ultralytics/pyrealsense2 AND rclpy, and joins the
same DDS domain as the robot driver. No build step; just run it.

Prereqs (other terminal): ros2 launch rebotarm_bringup bringup.launch.py ...

Run:
    .venv/bin/python TabletopSeg3D/3DDetection/scripts/perception_node.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import rclpy  # noqa: E402

from camera.realsense_capture import (  # noqa: E402
    build_runtime,
    enumerate_devices,
    get_aligned_frame_bundle,
    stop_runtimes,
)
from pipeline import (  # noqa: E402
    build_detection_3d,
    build_scene_point_cloud,
    estimate_table_normal,
    load_model,
    parse_class_set,
    parse_workspace,
    run_inference,
    smooth_normal,
)
from robot.detection_publisher import DetectionPublisher  # noqa: E402
from tracking import DetectionTracker, TrackerConfig  # noqa: E402


def find_latest_best_pt() -> str:
    cands = sorted((REPO_ROOT / "runs" / "segment").glob("*/weights/best.pt"), key=lambda p: p.stat().st_mtime)
    return str(cands[-1]) if cands else "yolo26m-seg.pt"


def find_latest_ee_T_cam() -> str:
    cands = sorted(
        (REPO_ROOT / "handeye" / "calibration_d405" / "captures").glob("session_*/ee_T_cam.json"),
        key=lambda p: p.stat().st_mtime,
    )
    return str(cands[-1]) if cands else ""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topic", default="/perception/detections", help="Output Detection3DArray topic.")
    ap.add_argument("--frame-id", default="base_link", help="Frame the published poses are expressed in.")
    ap.add_argument("--ee-t-cam", default=find_latest_ee_T_cam(), help="Path to ee_T_cam.json (default: latest).")
    ap.add_argument("--base-frame", default="base_link")
    ap.add_argument("--ee-frame", default="end_link")
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
    ap.add_argument("--target-class", default="")
    ap.add_argument("--min-depth", type=float, default=0.10)
    ap.add_argument("--max-depth", type=float, default=1.50)
    ap.add_argument("--min-points", type=int, default=500)
    ap.add_argument("--non-symmetric-classes", default="")
    ap.add_argument("--workspace", default="")
    ap.add_argument("--point-stride", type=int, default=2)
    ap.add_argument("--scene-max-points", type=int, default=80000)
    ap.add_argument("--warmup-frames", type=int, default=5)
    ap.add_argument("--table-normal-every", type=int, default=1)
    ap.add_argument("--tf-timeout", type=float, default=0.5)
    ap.add_argument("--log-every", type=float, default=1.0, help="Seconds between status prints (0 = silent).")
    args = ap.parse_args()
    args.workspace_bounds = parse_workspace(args.workspace)
    args.non_symmetric_set = parse_class_set(args.non_symmetric_classes)
    return args


def main() -> int:
    args = parse_args()
    print(f"[model] {args.model}")

    devices = enumerate_devices()
    if not devices:
        print("[error] no RealSense device found")
        return 1
    device = next((d for d in devices if d.serial_number == args.serial), devices[0])
    print(f"[realsense] using {device.name} ({device.serial_number})")
    model = load_model(args.model)
    runtime = build_runtime(device, args.width, args.height, args.fps)

    try:
        publisher = DetectionPublisher(
            args.ee_t_cam,
            topic=args.topic,
            frame_id=args.frame_id,
            base_frame=args.base_frame,
            ee_frame=args.ee_frame,
            tf_timeout=args.tf_timeout,
            node_name="perception_node",
        )
    except FileNotFoundError as exc:
        print(f"[error] {exc}")
        stop_runtimes([runtime])
        return 1
    print(f"[extrinsics] {args.ee_t_cam}")
    tracker = DetectionTracker(TrackerConfig())
    print(f"[ros] publishing vision_msgs/Detection3DArray on {args.topic} (frame {args.frame_id})")

    import open3d as o3d

    table_normal = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    for _ in range(args.warmup_frames):
        get_aligned_frame_bundle(runtime, args.min_depth, args.max_depth)

    frame_counter = 0
    last_log = 0.0
    try:
        while rclpy.ok():
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
            frame_counter += 1

            n_stable, n_pub, tf_ok, names = publisher.publish(tracked)

            if args.log_every > 0 and (now := time.perf_counter()) - last_log >= args.log_every:
                last_log = now
                if n_stable == 0:
                    print(f"[pub] frame {frame_counter}: 0 stable")
                elif not tf_ok:
                    print(f"[pub] frame {frame_counter}: {n_stable} stable but NO TF (published empty)")
                else:
                    print(f"[pub] frame {frame_counter}: {n_pub} in base ({', '.join(names)})")
    except KeyboardInterrupt:
        print("\n[ros] interrupted, shutting down")
    finally:
        stop_runtimes([runtime])
        publisher.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
