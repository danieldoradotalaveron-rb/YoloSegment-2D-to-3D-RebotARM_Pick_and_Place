#!/usr/bin/env python3
"""Realtime Open3D scene viewer with full-scene point cloud and 3D boxes.

Este archivo es solo el "wiring": parsea la CLI y ejecuta el bucle principal,
conectando las piezas que viven en módulos separados:
  - pipeline.py : núcleo bundle RGB-D -> detecciones 3D -> JSON (sin GUI).
  - viz_2d.py   : visualización 2D (OpenCV).
  - viz_3d.py   : visualización 3D (Open3D).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def find_latest_ee_T_cam() -> str:
    cands = sorted(
        (REPO_ROOT / "handeye" / "calibration_d405" / "captures").glob(
            "session_*/ee_T_cam.json"
        ),
        key=lambda p: p.stat().st_mtime,
    )
    return str(cands[-1]) if cands else ""


def to_display_frame(points_xyz: np.ndarray, transform: np.ndarray | None) -> np.ndarray:
    """Map camera-frame points to the display frame.

    With transform=None (plain demo) the cloud stays in the camera frame. With a 4x4
    base_T_cam (--real) the cloud is expressed in base_link, so an eye-in-hand camera
    moving around leaves the world static in the viewer. Works for an (N,3) cloud or
    the (8,3) box corners alike; None/empty passes through untouched.
    """
    if transform is None or points_xyz is None or len(points_xyz) == 0:
        return points_xyz
    pts = np.asarray(points_xyz, dtype=np.float64)
    return pts @ transform[:3, :3].T + transform[:3, 3]


def apply_base_workspace(detections, transform: np.ndarray | None, bounds) -> None:
    """Re-tag each detection's in_workspace using its center in the base frame.

    In --real the workspace is a fixed box in base_link (the robot's reachable table),
    not a sensor-relative volume, so the in/out test must use the base-frame center
    (filtered center mapped through base_T_cam). Overrides the camera-frame flag.
    """
    if bounds is None:
        return
    for det in detections:
        center = getattr(det, "center_filtered_xyz", None) or getattr(det, "center_xyz", None)
        if center is None:
            continue
        base_center = to_display_frame(np.asarray(center, dtype=np.float64), transform)
        det.in_workspace = center_in_workspace(base_center, bounds)

from camera.realsense_capture import (  # noqa: E402
    build_runtime,
    enumerate_devices,
    get_aligned_frame_bundle,
    select_serials,
    stop_runtimes,
)
from pipeline import (  # noqa: E402
    build_detection_3d,
    build_scene_point_cloud,
    center_in_workspace,
    estimate_table_normal,
    filtered_box_corners,
    frame_output_record,
    load_model,
    parse_class_set,
    parse_workspace,
    run_inference,
    smooth_normal,
)
from tracking import DetectionTracker, TrackerConfig  # noqa: E402
from viz_2d import compose_2d_view  # noqa: E402
from viz_3d import (  # noqa: E402
    BACKGROUND_COLOR_RGB,
    BACKGROUND_COLOR_RGBA,
    WORKSPACE_BOX_COLOR,
    build_legacy_point_cloud,
    color_for_detection,
    configure_view,
    dim_points_outside_workspace,
    highlight_object_points,
    scene_center,
    scene_eye,
    update_labels,
    update_line_set,
    workspace_box_corners,
)

VIEW2D_WINDOW = "Detections 2D"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime Open3D point-cloud scene viewer.")
    parser.add_argument("--list-devices", action="store_true", help="List connected RealSense devices and exit.")
    parser.add_argument("--serial", type=str, default="419522072950", help="RealSense serial to use.")
    parser.add_argument("--model", type=str, default="yolo26m-seg.pt", help="Ultralytics segmentation model.")
    parser.add_argument("--device", type=str, default="cpu", help="Inference device.")
    parser.add_argument("--width", type=int, default=640, help="Camera stream width.")
    parser.add_argument("--height", type=int, default=480, help="Camera stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Camera stream FPS.")
    parser.add_argument("--imgsz", type=int, default=448, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument(
        "--dedup-iou",
        type=float,
        default=0.7,
        help="Merge same-class detections whose mask IoU exceeds this (keep highest "
        "confidence), avoiding double 3D boxes box NMS misses. 0 disables.",
    )
    parser.add_argument("--max-det", type=int, default=10, help="Maximum detections per frame.")
    parser.add_argument("--target-class", type=str, default="", help="Optional class filter.")
    parser.add_argument("--min-depth", type=float, default=0.10, help="Minimum valid depth in meters.")
    parser.add_argument("--max-depth", type=float, default=1.50, help="Maximum valid depth in meters.")
    parser.add_argument("--min-points", type=int, default=500, help="Minimum object point count for a 3D box.")
    parser.add_argument("--non-symmetric-classes", type=str, default="", help="Comma-separated class names WITH a meaningful orientation; only these get a trusted yaw. Empty = none.")
    parser.add_argument("--symmetric-classes", type=str, default="", help="Comma-separated class names treated as symmetric (yaw free). Declarative; any class not in --non-symmetric-classes is symmetric.")
    parser.add_argument(
        "--workspace",
        type=str,
        default="",
        help="Actionable 3D volume (camera frame, meters) as "
        "'xmin,xmax,ymin,ymax,zmin,zmax'. Drawn as a box in the viewer, points "
        "outside are dimmed, and detections whose center falls outside are excluded "
        "from the JSON. Empty disables (whole scene is actionable).",
    )
    # Temporal tracker: smooths 3D pose over a sliding window and gates a `stable`
    # flag. Defaults mirror tracking.TrackerConfig; sense.yaml (track_*) overrides them.
    parser.add_argument("--track-window", type=int, default=10, help="Frames kept per track (sliding window).")
    parser.add_argument("--track-min-hits", type=int, default=6, help="Detections in window required for stable.")
    parser.add_argument("--track-max-misses", type=int, default=5, help="Frames unseen before a track is dropped.")
    parser.add_argument("--track-max-pos-std", type=float, default=0.008, help="Max center std (m) to be stable.")
    parser.add_argument("--track-max-yaw-std", type=float, default=10.0, help="Max yaw circular std (deg) to be stable.")
    parser.add_argument("--track-min-conf", type=float, default=0.8, help="Min mean confidence to be stable.")
    parser.add_argument("--track-assoc-dist", type=float, default=0.04, help="Association gate (m) for same-class match.")
    parser.add_argument("--track-stable-enter-frames", type=int, default=3, help="Consecutive frames meeting strict criteria before latching stable.")
    parser.add_argument("--track-stable-hysteresis", type=float, default=1.5, help="Exit thresholds = enter thresholds * this (looser to leave stable).")
    parser.add_argument("--track-stable-conf-margin", type=float, default=0.05, help="Exit stable when mean conf < (min_conf - this).")
    parser.add_argument("--warmup-frames", type=int, default=5, help="Camera warm-up frames.")
    parser.add_argument("--table-normal-every", type=int, default=1, help="Re-estimate the table normal every N frames (eye-in-hand). 0 = freeze at warm-up.")
    parser.add_argument("--frames", type=int, default=0, help="Run a fixed number of frames then exit.")
    parser.add_argument("--point-stride", type=int, default=2, help="Subsample stride for full-scene point cloud.")
    parser.add_argument("--scene-max-points", type=int, default=80000, help="Maximum full-scene points to keep after stride.")
    parser.add_argument("--show-object-points", action="store_true", help="Also color object mask points in the full-scene point cloud.")
    parser.add_argument("--show-labels", action="store_true", help="Show 3D labels for detected objects in the Open3D scene.")
    parser.add_argument("--raw-mode", action="store_true", help="Start the 3D viewer in raw lighting mode (simplified, no IBL/skybox).")
    parser.add_argument("--no-display", action="store_true", help="Disable Open3D window and print timing only.")
    parser.add_argument("--depth-filters", action="store_true", help="Enable RealSense depth post-processing (spatial+temporal).")
    parser.add_argument("--depth-preset", type=str, default="", help="RealSense visual preset (e.g. 'high_accuracy', 'high_density', 'default').")
    parser.add_argument("--show-2d", action="store_true", help="Show extra OpenCV window with RGB + YOLO masks overlay.")
    parser.add_argument("--view2d-size", type=str, default="1080x1920", help="2D window canvas size WxH (default 1080x1920 portrait).")
    parser.add_argument(
        "--view2d-layout",
        type=str,
        default="rgb-only",
        choices=["vertical", "horizontal", "rgb-only", "depth-only"],
        help="2D composition layout (default rgb-only).",
    )
    parser.add_argument(
        "--view2d-order",
        type=str,
        default="rgb-first",
        choices=["rgb-first", "depth-first"],
        help="Which panel goes first (top in vertical / left in horizontal).",
    )
    parser.add_argument("--view2d-fullscreen", action="store_true", help="Open the 2D window fullscreen (toggle in-window with 'f').")
    parser.add_argument("--view2d-depth-min", type=float, default=0.0, help="Manual min depth for 2D colormap (0 = auto percentile).")
    parser.add_argument("--view2d-depth-max", type=float, default=0.0, help="Manual max depth for 2D colormap (0 = auto percentile).")
    parser.add_argument("--view2d-pos", type=str, default="", help="Screen position of the 2D OpenCV window as X,Y (e.g. '0,960').")
    parser.add_argument("--view3d-size", type=str, default="", help="Open3D window size WxH (e.g. '1080x960').")
    parser.add_argument("--view3d-pos", type=str, default="", help="Open3D window position X,Y (e.g. '0,0').")
    # --real: same viewer, but ALSO publish stable detections in base_link to ROS so it can
    # run alongside RViz/hover_object (one process owns the D405). Without --real nothing
    # ROS is imported and the demo behaves exactly as before.
    parser.add_argument("--real", action="store_true", help="Also publish stable detections to ROS (vision_msgs/Detection3DArray).")
    parser.add_argument("--ee-t-cam", type=str, default="", help="Path to ee_T_cam.json for --real (default: latest session).")
    parser.add_argument("--topic", type=str, default="/perception/detections", help="Output topic for --real.")
    parser.add_argument("--ros-frame-id", type=str, default="base_link", help="Frame the published poses are expressed in (--real).")
    parser.add_argument("--base-frame", type=str, default="base_link", help="TF base frame (--real).")
    parser.add_argument("--ee-frame", type=str, default="end_link", help="TF end-effector frame the hand-eye was calibrated against (--real).")
    parser.add_argument("--tf-timeout", type=float, default=0.5, help="TF lookup timeout in seconds (--real).")
    parser.add_argument(
        "--workspace-base",
        type=str,
        default="",
        help="Actionable 3D volume in BASE_LINK (meters) as 'xmin,xmax,ymin,ymax,zmin,zmax'. "
        "Used only with --real: a fixed box on the robot's table (not sensor-relative). "
        "Drives the dimming, the drawn box and which detections are published. Empty = no gate.",
    )
    return parser.parse_args()


def parse_size_str(value: str, default_wh: tuple[int, int] = (1080, 1920)) -> tuple[int, int]:
    if not value:
        return default_wh
    try:
        parts = value.lower().replace("X", "x").split("x")
        w = max(64, int(parts[0]))
        h = max(64, int(parts[1]))
        return w, h
    except Exception:
        return default_wh


def parse_pos_str(value: str) -> tuple[int, int] | None:
    if not value:
        return None
    try:
        sep = "," if "," in value else "x"
        parts = value.split(sep)
        return int(parts[0]), int(parts[1])
    except Exception:
        return None


def print_connected_devices(devices: list[Any]) -> None:
    if not devices:
        print("No RealSense devices found.")
        return

    print("Connected RealSense devices:")
    for device in devices:
        print(f"- {device.name} | serial={device.serial_number}")


def main() -> int:
    args = parse_args()
    args.workspace_bounds = parse_workspace(getattr(args, "workspace", ""))
    if getattr(args, "workspace", "") and args.workspace_bounds is None:
        print(f"[workspace] ignoring invalid --workspace '{args.workspace}' "
              "(expected xmin,xmax,ymin,ymax,zmin,zmax)", file=sys.stderr)
    args.non_symmetric_set = parse_class_set(getattr(args, "non_symmetric_classes", ""))
    args.workspace_base_bounds = parse_workspace(getattr(args, "workspace_base", ""))
    import open3d as o3d

    devices = enumerate_devices()
    if args.list_devices:
        print_connected_devices(devices)
        return 0
    serial = select_serials(devices, [args.serial], expected_count=1)[0]
    device_map = {device.serial_number: device for device in devices}

    model = load_model(args.model)
    runtime = build_runtime(
        device_map[serial],
        args.width,
        args.height,
        args.fps,
        use_depth_filters=bool(args.depth_filters),
        visual_preset=args.depth_preset,
    )

    # --real: bring up the ROS publisher only on demand so the plain demo never needs
    # rclpy or a sourced ROS environment.
    detection_publisher = None
    if args.real:
        from robot.detection_publisher import DetectionPublisher

        ee_t_cam = args.ee_t_cam or find_latest_ee_T_cam()
        detection_publisher = DetectionPublisher(
            ee_t_cam,
            topic=args.topic,
            frame_id=args.ros_frame_id,
            base_frame=args.base_frame,
            ee_frame=args.ee_frame,
            tf_timeout=args.tf_timeout,
            node_name="tabletop_perception",
        )
        print(f"[real] extrinsics {ee_t_cam}")
        print(f"[real] publishing vision_msgs/Detection3DArray on {args.topic} (frame {args.ros_frame_id})")

    # --real renders in base_link so the world stays still while the eye-in-hand camera
    # moves. Wait briefly for the first base<-cam transform so the startup view is already
    # in base; if TF is not up yet, start in camera frame and switch once it arrives.
    render_base = detection_publisher is not None
    display_T = None
    # Workspace lives in the display frame: base_link with --real (a fixed box on the
    # robot's table), camera frame otherwise. In base mode we disable the camera-frame
    # gate in the pipeline and re-tag in_workspace from the base-frame center each frame.
    if render_base:
        active_bounds = args.workspace_base_bounds
        args.workspace_bounds = None
    else:
        active_bounds = args.workspace_bounds
    if render_base:
        for _ in range(20):
            display_T = detection_publisher.base_T_cam()
            if display_T is not None:
                break
            time.sleep(0.1)
        if display_T is None:
            print("[real] no TF yet; starting in camera frame, will switch to base when TF arrives")
        else:
            print("[real] rendering scene in base_link (world fixed, camera orbits)")

    frame_times: list[float] = []
    frame_counter = 0
    view2d_size = parse_size_str(getattr(args, "view2d_size", "") or "1080x1920")
    view2d_pos = parse_pos_str(getattr(args, "view2d_pos", "") or "")
    view2d_ready = False
    view2d_fullscreen_state = False
    view3d_size = parse_size_str(getattr(args, "view3d_size", "") or "", default_wh=(1280, 800))
    view3d_pos = parse_pos_str(getattr(args, "view3d_pos", "") or "")

    vis = None
    gui_app = None
    scene_pcd = o3d.geometry.PointCloud()
    scene_material = None
    box_sets: list[Any] = [o3d.geometry.LineSet() for _ in range(args.max_det)]
    box_added = [False for _ in range(args.max_det)]
    workspace_corners = workspace_box_corners(active_bounds)
    workspace_lineset = o3d.geometry.LineSet()
    if workspace_corners is not None:
        update_line_set(workspace_lineset, workspace_corners, WORKSPACE_BOX_COLOR, o3d)

    def box_visible(detection: Any) -> bool:
        return detection.box_corners_xyz is not None and getattr(detection, "in_workspace", True)

    center_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.08)
    label_mode = bool(args.show_labels and not args.no_display)
    window_closed = False
    table_normal = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    try:
        for _ in range(args.warmup_frames):
            get_aligned_frame_bundle(runtime, args.min_depth, args.max_depth)

        warm_bundle = get_aligned_frame_bundle(runtime, args.min_depth, args.max_depth)
        model.predict(
            source=warm_bundle["color"],
            task="segment",
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            verbose=False,
        )

        initial_scene_points, initial_scene_colors = build_scene_point_cloud(
            color_image=warm_bundle["color"],
            depth_m=warm_bundle["depth"].astype(np.float32) * float(warm_bundle["depth_scale"]),
            intrinsics=warm_bundle["aligned_intrinsics"],
            args=args,
        )
        table_normal = estimate_table_normal(initial_scene_points, o3d)
        initial_depth_m = warm_bundle["depth"].astype(np.float32) * float(warm_bundle["depth_scale"])
        initial_detections = run_inference(model, warm_bundle["color"], args)
        initial_detections_3d = [
            build_detection_3d(det, initial_depth_m, warm_bundle["aligned_intrinsics"], table_normal, args)
            for det in initial_detections
        ]
        # One tracker for the whole session. .
        tracker = DetectionTracker(
            TrackerConfig(
                window_size=args.track_window,
                min_hits=args.track_min_hits,
                max_misses=args.track_max_misses,
                max_position_std_m=args.track_max_pos_std,
                max_yaw_std_deg=args.track_max_yaw_std,
                min_confidence=args.track_min_conf,
                assoc_max_dist_m=args.track_assoc_dist,
                stable_enter_frames=args.track_stable_enter_frames,
                stable_hysteresis=args.track_stable_hysteresis,
                stable_conf_margin=args.track_stable_conf_margin,
            )
        )
        initial_detections_3d = tracker.update(initial_detections_3d)
        if render_base:
            apply_base_workspace(initial_detections_3d, display_T, active_bounds)

        # Display-frame copy (identity in camera mode, base_T_cam in --real). The cloud is
        # mapped to the display frame; the workspace box + bounds already live in that frame.
        initial_disp_points = to_display_frame(initial_scene_points, display_T)
        initial_scene_colors = dim_points_outside_workspace(initial_disp_points, initial_scene_colors, active_bounds)
        if args.show_object_points:
            initial_scene_colors = highlight_object_points(initial_scene_points, initial_scene_colors, initial_detections_3d)

        if not args.no_display:
            if label_mode:
                import open3d.visualization.gui as gui
                import open3d.visualization.rendering as rendering

                gui_app = gui.Application.instance
                gui_app.initialize()
                vis = o3d.visualization.O3DVisualizer("Realtime Open3D Scene", view3d_size[0], view3d_size[1])
                if view3d_pos is not None:
                    try:
                        vis.os_frame = (view3d_pos[0], view3d_pos[1], view3d_size[0], view3d_size[1])
                    except Exception:
                        pass
                vis.show_axes = True
                vis.show_settings = False
                vis.show_ground = False
                vis.show_skybox(False)
                if getattr(args, "raw_mode", False):
                    # Same as the settings-panel "raw mode": simplified lighting env.
                    vis.enable_raw_mode(True)
                # Set the background AFTER raw mode so the chosen color is not reset
                # when the simplified lighting environment is applied.
                vis.set_background(BACKGROUND_COLOR_RGBA, None)

                def on_window_close() -> bool:
                    nonlocal window_closed
                    window_closed = True
                    return True

                vis.set_on_close(on_window_close)

                scene_material = rendering.MaterialRecord()
                scene_material.shader = "defaultUnlit"
                scene_material.point_size = 5.0

                vis.add_geometry(
                    "scene",
                    build_legacy_point_cloud(o3d, initial_disp_points, initial_scene_colors),
                    scene_material,
                )

                if workspace_corners is not None:
                    vis.add_geometry("workspace", workspace_lineset)

                initial_corners = [
                    to_display_frame(filtered_box_corners(det, table_normal), display_T)
                    for det in initial_detections_3d
                ]
                for idx, detection in enumerate(initial_detections_3d):
                    if idx >= args.max_det or not box_visible(detection):
                        continue
                    update_line_set(
                        box_sets[idx],
                        initial_corners[idx],
                        color_for_detection(detection, idx),
                        o3d,
                    )
                    vis.add_geometry(f"box_{idx}", box_sets[idx])
                    box_added[idx] = True

                update_labels(vis, initial_detections_3d, initial_corners)
                gui_app.add_window(vis)
                vis.setup_camera(
                    45.0,
                    scene_center(initial_disp_points).astype(np.float32),
                    scene_eye(initial_disp_points, scene_center(initial_disp_points)).astype(np.float32),
                    np.array([0.0, -1.0, 0.0], dtype=np.float32),
                )
                gui_app.run_one_tick()
            else:
                vis = o3d.visualization.Visualizer()
                if view3d_pos is not None:
                    vis.create_window(
                        window_name="Realtime Open3D Scene",
                        width=view3d_size[0],
                        height=view3d_size[1],
                        left=int(view3d_pos[0]),
                        top=int(view3d_pos[1]),
                    )
                else:
                    vis.create_window(
                        window_name="Realtime Open3D Scene",
                        width=view3d_size[0],
                        height=view3d_size[1],
                    )
                scene_pcd.points = o3d.utility.Vector3dVector(initial_disp_points.astype(np.float64))
                scene_pcd.colors = o3d.utility.Vector3dVector(initial_scene_colors.astype(np.float64))
                vis.add_geometry(scene_pcd)

                if workspace_corners is not None:
                    vis.add_geometry(workspace_lineset, reset_bounding_box=False)

                first_center = None
                for idx, detection in enumerate(initial_detections_3d):
                    if idx >= len(box_sets):
                        break
                    if not box_visible(detection):
                        continue
                    update_line_set(
                        box_sets[idx],
                        to_display_frame(filtered_box_corners(detection, table_normal), display_T),
                        color_for_detection(detection, idx),
                        o3d,
                    )
                    vis.add_geometry(box_sets[idx], reset_bounding_box=False)
                    box_added[idx] = True
                    if first_center is None and detection.center_xyz is not None:
                        first_center = to_display_frame(
                            np.array(detection.center_xyz, dtype=np.float64), display_T
                        )

                if first_center is not None:
                    center_frame.translate(first_center, relative=True)
                vis.add_geometry(center_frame)
                render_option = vis.get_render_option()
                render_option.background_color = BACKGROUND_COLOR_RGB
                render_option.point_size = 2.0
                configure_view(vis, scene_center(initial_disp_points))

        while True:
            loop_start = time.perf_counter()
            bundle = get_aligned_frame_bundle(runtime, args.min_depth, args.max_depth)
            depth_m = bundle["depth"].astype(np.float32) * float(bundle["depth_scale"])

            infer_start = time.perf_counter()
            detections = run_inference(model, bundle["color"], args)
            infer_ms = (time.perf_counter() - infer_start) * 1000.0

            geom_start = time.perf_counter()
            scene_points, scene_colors = build_scene_point_cloud(
                color_image=bundle["color"],
                depth_m=depth_m,
                intrinsics=bundle["aligned_intrinsics"],
                args=args,
            )
            # Eye-in-hand: the table reorients in the camera frame as the arm moves, so
            # refresh the table normal from the live scene (every N frames, EMA-smoothed)
            # rather than trusting the warm-up estimate. table_normal_every=0 keeps it frozen.
            if args.table_normal_every > 0 and frame_counter % args.table_normal_every == 0:
                table_normal = smooth_normal(table_normal, estimate_table_normal(scene_points, o3d))
            detections_3d = [
                build_detection_3d(det, depth_m, bundle["aligned_intrinsics"], table_normal, args)
                for det in detections
            ]
            tracked_detections = tracker.update(detections_3d)

            # Refresh base<-cam and re-tag in_workspace from the base-frame center BEFORE
            # publishing, so the published/picked set matches the base-frame workspace.
            if render_base:
                display_T = detection_publisher.base_T_cam()
                apply_base_workspace(tracked_detections, display_T, active_bounds)

            if detection_publisher is not None:
                n_stable, n_pub, tf_ok, names = detection_publisher.publish(tracked_detections)
                if frame_counter % 30 == 0:
                    if n_stable == 0:
                        print(f"[real] frame {frame_counter}: 0 stable")
                    elif not tf_ok:
                        print(f"[real] frame {frame_counter}: {n_stable} stable but NO TF (published empty)")
                    else:
                        print(f"[real] frame {frame_counter}: {n_pub} in base ({', '.join(names)})")

            # Render the cloud in the display frame (base_link with --real); the workspace
            # box + dimming bounds already live in that frame.
            disp_points = to_display_frame(scene_points, display_T)
            scene_colors = dim_points_outside_workspace(disp_points, scene_colors, active_bounds)
            if args.show_object_points:
                scene_colors = highlight_object_points(scene_points, scene_colors, tracked_detections)
            geom_ms = (time.perf_counter() - geom_start) * 1000.0

            loop_time = time.perf_counter() - loop_start
            frame_times.append(loop_time)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps_value = len(frame_times) / sum(frame_times)

            if args.show_2d and not args.no_display:
                import cv2

                if not view2d_ready:
                    cv2.namedWindow(VIEW2D_WINDOW, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(VIEW2D_WINDOW, view2d_size[0], view2d_size[1])
                    if view2d_pos is not None:
                        try:
                            cv2.moveWindow(VIEW2D_WINDOW, int(view2d_pos[0]), int(view2d_pos[1]))
                        except Exception:
                            pass
                    if args.view2d_fullscreen:
                        cv2.setWindowProperty(VIEW2D_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        view2d_fullscreen_state = True
                    view2d_ready = True

                composite = compose_2d_view(
                    color_bgr=bundle["color"],
                    depth_m=depth_m,
                    detections=detections,
                    fps=fps_value,
                    layout=args.view2d_layout,
                    order=args.view2d_order,
                    size_wh=view2d_size,
                    depth_manual_lo=float(args.view2d_depth_min),
                    depth_manual_hi=float(args.view2d_depth_max),
                )
                cv2.imshow(VIEW2D_WINDOW, composite)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("f"):
                    view2d_fullscreen_state = not view2d_fullscreen_state
                    cv2.setWindowProperty(
                        VIEW2D_WINDOW,
                        cv2.WND_PROP_FULLSCREEN,
                        cv2.WINDOW_FULLSCREEN if view2d_fullscreen_state else cv2.WINDOW_NORMAL,
                    )
                    if not view2d_fullscreen_state:
                        cv2.resizeWindow(VIEW2D_WINDOW, view2d_size[0], view2d_size[1])

            if args.no_display:
                print(
                    json.dumps(
                        frame_output_record(
                            frame_index=frame_counter,
                            fps_value=fps_value,
                            infer_ms=infer_ms,
                            geom_ms=geom_ms,
                            scene_points=scene_points,
                            table_normal=table_normal,
                            detections_3d=tracked_detections,
                        ),
                        ensure_ascii=False,
                    )
                )
            else:
                if label_mode:
                    vis.remove_geometry("scene")
                    vis.add_geometry(
                        "scene",
                        build_legacy_point_cloud(o3d, disp_points, scene_colors),
                        scene_material,
                    )

                    # Filtered (steady) corners, computed once and reused for both the
                    # box geometry and the label anchor so the text stops jumping.
                    draw_corners = [
                        to_display_frame(filtered_box_corners(det, table_normal), display_T)
                        for det in tracked_detections
                    ]
                    for idx in range(args.max_det):
                        box_name = f"box_{idx}"
                        if idx < len(tracked_detections) and box_visible(tracked_detections[idx]):
                            update_line_set(
                                box_sets[idx],
                                draw_corners[idx],
                                color_for_detection(tracked_detections[idx], idx),
                                o3d,
                            )
                            if box_added[idx]:
                                vis.remove_geometry(box_name)
                            vis.add_geometry(box_name, box_sets[idx])
                            box_added[idx] = True
                        elif box_added[idx]:
                            vis.remove_geometry(box_name)
                            box_added[idx] = False

                    update_labels(vis, tracked_detections, draw_corners)
                    vis.post_redraw()
                    if window_closed or not gui_app.run_one_tick():
                        break
                else:
                    scene_pcd.points = o3d.utility.Vector3dVector(disp_points.astype(np.float64))
                    scene_pcd.colors = o3d.utility.Vector3dVector(scene_colors.astype(np.float64))
                    vis.update_geometry(scene_pcd)

                    first_center = None
                    for idx, box_set in enumerate(box_sets):
                        if idx < len(tracked_detections):
                            detection = tracked_detections[idx]
                            if box_visible(detection):
                                update_line_set(
                                    box_set,
                                    to_display_frame(filtered_box_corners(detection, table_normal), display_T),
                                    color_for_detection(detection, idx),
                                    o3d,
                                )
                                if not box_added[idx]:
                                    vis.add_geometry(box_set, reset_bounding_box=False)
                                    box_added[idx] = True
                            elif box_added[idx]:
                                vis.remove_geometry(box_set, reset_bounding_box=False)
                                box_added[idx] = False
                            if first_center is None and detection.center_xyz is not None:
                                first_center = to_display_frame(
                                    np.array(detection.center_xyz, dtype=np.float64), display_T
                                )
                        else:
                            if box_added[idx]:
                                vis.remove_geometry(box_set, reset_bounding_box=False)
                                box_added[idx] = False

                        if box_added[idx]:
                            vis.update_geometry(box_set)

                    if first_center is None:
                        center_frame.translate(-np.asarray(center_frame.get_center()), relative=True)
                    else:
                        center_frame.translate(first_center - np.asarray(center_frame.get_center()), relative=True)
                    vis.update_geometry(center_frame)

                    if not vis.poll_events():
                        break
                    vis.update_renderer()

            frame_counter += 1
            if args.frames > 0 and frame_counter >= args.frames:
                break

    finally:
        if vis is not None:
            if hasattr(vis, "destroy_window"):
                vis.destroy_window()
            elif hasattr(vis, "close"):
                vis.close()
        if gui_app is not None and hasattr(gui_app, "quit"):
            gui_app.quit()
        if args.show_2d and not args.no_display:
            try:
                import cv2

                cv2.destroyAllWindows()
            except Exception:
                pass
        stop_runtimes([runtime])
        if detection_publisher is not None:
            detection_publisher.shutdown()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
