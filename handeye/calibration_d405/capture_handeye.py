"""Interactive eye-in-hand calibration capture.

Reads the live RealSense color frame (via the project's realsense_capture module)
and the robot pose `base_T_ee` from TF (base_link -> end_link), and lets you save
synchronized samples while you move the arm by hand (gravity compensation).

Each sample stores the raw color image plus `base_T_ee`; ChArUco detection and
the actual hand-eye solve happen offline in `calibrate_handeye.py`, so the saved
session is the single source of truth and can be re-solved without the robot.

Prereqs (in another terminal):
    ros2 launch rebotarm_bringup bringup.launch.py channel:=/dev/ttyACM0
    ros2 service call /rebotarm/enable std_srvs/srv/Trigger
    ros2 service call /rebotarm/gravity_compensation/start std_srvs/srv/Trigger

Run (must use the repo venv so rclpy + pyrealsense2 + cv2>=4.7 are all available):
    .venv/bin/python handeye/calibration_d405/capture_handeye.py

Controls (focus the preview window):
    c / SPACE : capture a sample (needs a valid board view + fresh TF)
    u         : undo last sample
    q / ESC   : finish and write the session
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
HANDEYE_DIR = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT / "TabletopSeg3D/3DDetection/src"))
sys.path.insert(0, str(HANDEYE_DIR))

from camera.realsense_capture import (  # noqa: E402
    build_runtime,
    enumerate_devices,
    get_aligned_frame_bundle,
    stop_runtimes,
)
from charuco_utils import detect, draw_overlay, estimate_pose, intrinsics_to_K, load_board  # noqa: E402
from robot.robot_pose import RobotPoseReader  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", default=str(HANDEYE_DIR / "charuco_boox_go_103.json"))
    ap.add_argument("--out-dir", default=str(HANDEYE_DIR / "captures"))
    ap.add_argument("--append", default="", help="Resume an existing session dir/json: keep its samples and add more.")
    ap.add_argument("--base-frame", default="base_link")
    ap.add_argument("--ee-frame", default="end_link")
    ap.add_argument("--serial", default="", help="RealSense serial (default: first device found).")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--min-corners", type=int, default=6, help="Min ChArUco corners to allow a capture.")
    ap.add_argument("--capture-frames", type=int, default=10, help="Frames polled per 'c' press; best is kept.")
    ap.add_argument("--tf-timeout", type=float, default=0.5, help="Seconds to wait for a TF lookup.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    spec = load_board(args.board)
    print(
        f"[board] {spec.squares_x}x{spec.squares_y}, square={spec.square_length_m*1000:.2f}mm, "
        f"marker={spec.marker_length_m*1000:.2f}mm"
    )

    devices = enumerate_devices()
    if not devices:
        print("[realsense] no device found")
        return 1
    device = next((d for d in devices if d.serial_number == args.serial), devices[0])
    print(f"[realsense] using {device.name} ({device.serial_number})")
    runtime = build_runtime(device, args.width, args.height, args.fps, enable_depth=True, enable_color=True)

    pose_reader = RobotPoseReader(args.base_frame, args.ee_frame, node_name="handeye_capture")
    print(f"[tf] reading {args.base_frame} -> {args.ee_frame}")

    samples: list[dict] = []
    intrinsics_saved: dict | None = None
    if args.append:
        sess_path = Path(args.append)
        sess_path = sess_path / "session.json" if sess_path.is_dir() else sess_path
        out_dir = sess_path.parent
        existing = json.loads(sess_path.read_text())
        samples = existing["samples"]
        intrinsics_saved = existing["intrinsics"]
        print(f"[append] resuming {sess_path} with {len(samples)} existing samples")
        if int(intrinsics_saved.get("width", args.width)) != args.width:
            print(
                f"[append] WARNING: existing intrinsics width={intrinsics_saved.get('width')} "
                f"!= --width {args.width}. Use the same resolution as the original session."
            )
    else:
        session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        out_dir = Path(args.out_dir) / session_name
        out_dir.mkdir(parents=True, exist_ok=True)

    next_index = (max((s["index"] for s in samples), default=-1) + 1)
    status_msg = "move arm, then press c"
    status_color = (255, 255, 255)
    shape_printed = False
    window = "hand-eye capture (c=capture  u=undo  s=save-debug  q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    session_path = out_dir / "session.json"

    def persist_session() -> None:
        """Write session.json with whatever we have so far. Called after every
        capture (and on exit) so Ctrl+C never loses the recorded poses."""
        if not samples or intrinsics_saved is None:
            return
        session = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "base_frame": args.base_frame,
            "ee_frame": args.ee_frame,
            "board_json": str(Path(args.board).resolve()),
            "intrinsics": intrinsics_saved,
            "samples": samples,
        }
        session_path.write_text(json.dumps(session, indent=2))

    try:
        while True:
            bundle = get_aligned_frame_bundle(runtime, depth_min_m=0.0, depth_max_m=10.0)
            color = bundle["color"]
            intr = bundle["aligned_intrinsics"]
            if intrinsics_saved is None:
                intrinsics_saved = intr
            if not shape_printed:
                print(f"[realsense] color frame shape = {color.shape} (h, w, c)")
                shape_printed = True
            K, dist = intrinsics_to_K(intr)

            det = detect(spec, color)
            pose = estimate_pose(spec, det, K, dist, min_corners=args.min_corners)
            preview = draw_overlay(spec, color, det)

            valid_board = pose is not None
            color_txt = (0, 200, 0) if valid_board else (0, 0, 255)
            cv2.putText(
                preview,
                f"corners={det.n_corners}/{args.min_corners}  board={'OK' if valid_board else 'weak'}  samples={len(samples)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color_txt,
                2,
            )
            cv2.putText(preview, status_msg, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("u"), ord("U")) and samples:
                removed = samples.pop()
                Path(removed["image"]).unlink(missing_ok=True)
                persist_session()
                status_msg, status_color = f"undo -> {len(samples)} samples", (0, 200, 255)
                print(f"[undo] removed sample {removed['index']} (now {len(samples)})")
                continue
            if key in (ord("s"), ord("S")):
                dbg = out_dir / "debug_frame.png"
                cv2.imwrite(str(dbg), color)
                n_markers = 0 if det.marker_ids is None else len(det.marker_ids)
                status_msg, status_color = f"saved debug frame ({n_markers} markers)", (0, 200, 255)
                print(f"[debug] wrote {dbg}  markers={n_markers}  corners={det.n_corners}  shape={color.shape}")
                continue
            if key in (ord("c"), ord("C"), ord(" ")):
                # Poll several frames and keep the one with the most ChArUco corners,
                # so detection flicker at the instant of pressing does not lose a pose.
                best_color, best_corners = None, -1
                for _ in range(args.capture_frames):
                    b = get_aligned_frame_bundle(runtime, depth_min_m=0.0, depth_max_m=10.0)
                    c = b["color"]
                    Kc, distc = intrinsics_to_K(b["aligned_intrinsics"])
                    dc = detect(spec, c)
                    pc = estimate_pose(spec, dc, Kc, distc, min_corners=args.min_corners)
                    if pc is not None and dc.n_corners > best_corners:
                        best_color, best_corners = c, dc.n_corners
                if best_color is None:
                    status_msg, status_color = f"REJECTED: weak board (best corners<{args.min_corners})", (0, 0, 255)
                    print("[capture] rejected: weak/absent board view across polled frames")
                    continue
                base_T_ee = pose_reader.lookup(args.tf_timeout)
                if base_T_ee is None:
                    status_msg, status_color = "REJECTED: no TF base_T_ee", (0, 0, 255)
                    print("[capture] rejected: no TF for base_T_ee")
                    continue
                idx = next_index
                next_index += 1
                img_path = out_dir / f"color_{idx:04d}.png"
                cv2.imwrite(str(img_path), best_color)
                samples.append(
                    {
                        "index": idx,
                        "image": str(img_path),
                        "base_T_ee": base_T_ee.tolist(),
                        "n_corners": int(best_corners),
                    }
                )
                persist_session()
                status_msg, status_color = f"CAPTURED #{idx} (corners={best_corners}, saved)", (0, 200, 0)
                print(
                    f"[capture] sample {idx}: corners={best_corners}, "
                    f"ee_pos={np.round(base_T_ee[:3, 3], 3).tolist()} -> session saved"
                )
    except KeyboardInterrupt:
        print("\n[session] Ctrl+C: session already saved incrementally, exiting cleanly")
    finally:
        persist_session()
        cv2.destroyAllWindows()
        stop_runtimes([runtime])
        pose_reader.shutdown()

    if not samples:
        print("[session] no samples captured; nothing written")
        return 1
    print(f"[session] {len(samples)} samples -> {session_path}")
    if len(samples) < 10:
        print(f"[session] WARNING: {len(samples)} samples is low; 12-20 varied poses give a better solve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
