"""Offline RealSense capture: RGB frames or RGBD bundles."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMERA_SRC = REPO_ROOT / "TabletopSeg3D" / "3DDetection" / "src"
if str(CAMERA_SRC) not in sys.path:
    sys.path.insert(0, str(CAMERA_SRC))

from camera.realsense_capture import (  # noqa: E402
    _apply_visual_preset,
    build_depth_filters,
)

CAPTURE_ROOT = Path("dataset_capture")
RGB_DIR = CAPTURE_ROOT / "rgb"
RGBD_DIR = CAPTURE_ROOT / "rgbd"
RGBD_BG_DIR = CAPTURE_ROOT / "rgbd_backgrounds"

WIDTH = 640
HEIGHT = 480
FPS = 30

WINDOW_RGB = "RealSense RGB capture"
WINDOW_RGBD = "RealSense RGBD capture"

CAPTURE_ID_RE = re.compile(r"^capture_(\d+)$")

# Logitech presenter (046d:c540) sends Left/Right arrow keys.
KEY_LEFT = frozenset({65361, 2424832})  # 0xFF51 / 0x250000
KEY_RIGHT = frozenset({65363, 2555904})  # 0xFF53 / 0x270000

RGB_HINT = "centro rueda o flechas mando = guardar | q: salir"
RGBD_HINT = [
    "-> / centro rueda: RGBD objeto (rgbd/)",
    "<- : RGBD fondo vacio (rgbd_backgrounds/<ultimo id>/)",
    "q: salir",
]


def intrinsics_to_dict(intrinsics: rs.intrinsics) -> dict[str, Any]:
    return {
        "width": intrinsics.width,
        "height": intrinsics.height,
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "ppx": intrinsics.ppx,
        "ppy": intrinsics.ppy,
        "model": str(intrinsics.model),
        "coeffs": list(intrinsics.coeffs),
    }


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def next_rgbd_capture_dir(root: Path = RGBD_DIR) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    highest = 0
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = CAPTURE_ID_RE.match(child.name)
        if match:
            highest = max(highest, int(match.group(1)))
    return root / f"capture_{highest + 1:06d}"


def background_exists(capture_id: str) -> bool:
    bg_dir = RGBD_BG_DIR / capture_id
    return bg_dir.is_dir() and (bg_dir / "rgb.png").is_file()


def pending_background_capture_id() -> str | None:
    """Lowest capture_XXXXXX in rgbd/ that has no paired background yet."""
    if not RGBD_DIR.is_dir():
        return None
    missing: list[int] = []
    for child in RGBD_DIR.iterdir():
        if not child.is_dir():
            continue
        match = CAPTURE_ID_RE.match(child.name)
        if not match or not (child / "rgb.png").is_file():
            continue
        if not background_exists(child.name):
            missing.append(int(match.group(1)))
    if not missing:
        return None
    return f"capture_{min(missing):06d}"


def rgbd_overlay_hints(*, pending: str | None, filter_hint: str) -> list[str]:
    if pending:
        return [
            f"PENDIENTE: falta bg de {pending} — pulsa <- (escena vacia)",
            "-> bloqueado hasta guardar el fondo",
            f"q: salir{filter_hint}",
        ]
    lines = list(RGBD_HINT)
    lines[-1] = lines[-1] + filter_hint
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline RealSense capture.")
    parser.add_argument(
        "--mode",
        choices=("rgb", "rgbd"),
        default="rgb",
        help="rgb: save .jpg frames under dataset_capture/rgb/; "
        "rgbd: save rgb.png + depth.npy + intrinsics.yaml + meta.yaml under dataset_capture/rgbd/",
    )
    parser.add_argument("--rgbd", action="store_true", help="Shortcut for --mode rgbd.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="RGBD: replace an existing rgbd_backgrounds/<capture_id> folder.",
    )
    parser.add_argument("--serial", default="", help="RealSense device serial (default: first device).")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument(
        "--depth-filters",
        action="store_true",
        help="RGBD only: RealSense spatial+temporal depth filters (hole fill, smoother depth).",
    )
    parser.add_argument(
        "--depth-preset",
        default="",
        help="RGBD only: RealSense visual preset (e.g. high_density, high_accuracy).",
    )
    return parser.parse_args()


def device_info(device: rs.device) -> dict[str, str]:
    def safe(info_key: rs.camera_info) -> str:
        try:
            return device.get_info(info_key)
        except RuntimeError:
            return ""

    return {
        "serial": safe(rs.camera_info.serial_number),
        "device_name": safe(rs.camera_info.name),
        "firmware_version": safe(rs.camera_info.firmware_version),
    }


def start_pipeline(
    args: argparse.Namespace, *, enable_depth: bool
) -> tuple[rs.pipeline, rs.align | None, float, dict[str, str], list | None, str | None]:
    ctx = rs.context()
    devices = list(ctx.query_devices())
    if not devices:
        raise RuntimeError("No RealSense devices detected.")

    serial = args.serial.strip()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    else:
        serial = devices[0].get_info(rs.camera_info.serial_number)
        config.enable_device(serial)

    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    align = None
    depth_scale = 1.0
    depth_filters = None
    applied_preset = None
    if enable_depth:
        config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    pipeline = rs.pipeline()
    profile = pipeline.start(config)
    info = device_info(profile.get_device())

    if enable_depth:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        align = rs.align(rs.stream.color)
        preset = getattr(args, "depth_preset", "") or ""
        applied_preset = _apply_visual_preset(depth_sensor, preset)
        if getattr(args, "depth_filters", False):
            depth_filters = build_depth_filters()
            print("[capture] depth filters ON (spatial + temporal + hole fill)")

    return pipeline, align, depth_scale, info, depth_filters, applied_preset


def apply_depth_filters(depth_frame: rs.depth_frame, filters: list | None) -> rs.depth_frame:
    if not filters:
        return depth_frame
    filtered = depth_frame
    for depth_filter in filters:
        filtered = depth_filter.process(filtered)
    return filtered


def write_rgbd_bundle(out_dir: Path, bundle: dict[str, Any], meta: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "rgb.png"), bundle["color"])
    np.save(out_dir / "depth.npy", bundle["depth_m"])
    write_yaml(out_dir / "intrinsics.yaml", bundle["intrinsics"])
    write_yaml(out_dir / "meta.yaml", meta)


def poll_window_key(*, rgbd: bool) -> str:
    """Return 'quit', 'save'|'object'|'background', or '' (window needs focus)."""
    key = cv2.waitKeyEx(1)
    if key < 0:
        return ""
    # Arrow keys first: 65361 (Left) has low byte 0x51 == ord('Q') — must not mask with & 0xFF.
    if key in KEY_LEFT:
        return "background" if rgbd else "save"
    if key in KEY_RIGHT:
        return "object" if rgbd else "save"
    if key in (ord("q"), ord("Q")):
        return "quit"
    return ""


def overlay_hint(preview: np.ndarray, lines: list[str]) -> None:
    y = 28
    for line in lines:
        cv2.putText(
            preview,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
        )
        y += 26


def run_rgb(args: argparse.Namespace) -> None:
    RGB_DIR.mkdir(parents=True, exist_ok=True)
    pipeline, _, _, _, _, _ = start_pipeline(args, enable_depth=False)

    state: dict[str, Any] = {"frame": None, "counter": 0}

    def save_frame() -> None:
        frame = state["frame"]
        if frame is None:
            return
        while True:
            filename = RGB_DIR / f"frame_{int(time.time())}_{state['counter']:04d}.jpg"
            state["counter"] += 1
            if not filename.exists():
                break
        cv2.imwrite(str(filename), frame)
        print(f"saved: {filename}")

    def on_mouse(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_MBUTTONDOWN:
            save_frame()

    cv2.namedWindow(WINDOW_RGB)
    cv2.setMouseCallback(WINDOW_RGB, on_mouse)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            state["frame"] = color
            preview = color.copy()
            overlay_hint(preview, [RGB_HINT])
            cv2.imshow(WINDOW_RGB, preview)
            action = poll_window_key(rgbd=False)
            if action == "save":
                save_frame()
            elif action == "quit":
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def run_rgbd(args: argparse.Namespace) -> None:
    RGBD_DIR.mkdir(parents=True, exist_ok=True)
    RGBD_BG_DIR.mkdir(parents=True, exist_ok=True)

    pipeline, align, depth_scale, info, depth_filters, applied_preset = start_pipeline(
        args, enable_depth=True
    )
    assert align is not None

    state: dict[str, Any] = {"bundle": None}

    pending = pending_background_capture_id()
    if pending:
        print(
            f"[capture] PENDIENTE: falta background de {pending}. "
            "-> bloqueado; pulsa <- con escena vacia."
        )

    def save_object() -> None:
        bundle = state["bundle"]
        if bundle is None:
            return
        pending_id = pending_background_capture_id()
        if pending_id:
            print(
                f"SKIP object: falta background de {pending_id} "
                f"(pulsa <- con escena vacia antes de otra captura)"
            )
            return
        out_dir = next_rgbd_capture_dir(RGBD_DIR)
        meta = dict(bundle["meta"])
        meta["capture_id"] = out_dir.name
        meta["kind"] = "object"
        write_rgbd_bundle(out_dir, bundle, meta)
        print(f"saved object: {out_dir}/")
        print(f"  -> ahora pulsa <- (escena vacia) para bg de {out_dir.name}")

    def save_background() -> None:
        bundle = state["bundle"]
        if bundle is None:
            return
        capture_id = pending_background_capture_id()
        if capture_id is None:
            print("SKIP background: todos los objetos tienen fondo (pulsa -> para nueva captura)")
            return
        out_dir = RGBD_BG_DIR / capture_id
        if out_dir.exists() and not args.overwrite:
            print(
                f"SKIP background: {out_dir} already exists (pass --overwrite to replace)"
            )
            return
        meta = dict(bundle["meta"])
        meta["capture_id"] = capture_id
        meta["kind"] = "background"
        meta["for_capture"] = capture_id
        write_rgbd_bundle(out_dir, bundle, meta)
        print(f"saved background: {out_dir}/")

    def on_mouse(event, x, y, flags, param) -> None:
        if event == cv2.EVENT_MBUTTONDOWN:
            save_object()

    cv2.namedWindow(WINDOW_RGBD)
    cv2.setMouseCallback(WINDOW_RGBD, on_mouse)

    filter_hint = " | depth-filters ON" if depth_filters else ""
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            depth_frame = apply_depth_filters(depth_frame, depth_filters)
            color = np.asanyarray(color_frame.get_data())
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_m = depth_raw.astype(np.float32) * depth_scale

            intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
            state["bundle"] = {
                "color": color,
                "depth_m": depth_m,
                "intrinsics": intrinsics_to_dict(intrinsics),
                "meta": {
                    "timestamp_unix": time.time(),
                    "timestamp_ms": frames.get_timestamp(),
                    "frame_number": frames.get_frame_number(),
                    "serial": info["serial"],
                    "device_name": info["device_name"],
                    "firmware_version": info["firmware_version"],
                    "width": args.width,
                    "height": args.height,
                    "fps": args.fps,
                    "depth_scale": depth_scale,
                    "depth_units": "meters",
                    "aligned_to": "color",
                    "depth_filters": bool(depth_filters),
                    "depth_preset": applied_preset or "",
                },
            }

            preview = color.copy()
            overlay_hint(
                preview,
                rgbd_overlay_hints(
                    pending=pending_background_capture_id(),
                    filter_hint=filter_hint,
                ),
            )
            cv2.imshow(WINDOW_RGBD, preview)
            action = poll_window_key(rgbd=True)
            if action == "object":
                save_object()
            elif action == "background":
                save_background()
            elif action == "quit":
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if args.rgbd:
        args.mode = "rgbd"
    if args.mode == "rgbd":
        run_rgbd(args)
    else:
        run_rgb(args)


if __name__ == "__main__":
    main()
