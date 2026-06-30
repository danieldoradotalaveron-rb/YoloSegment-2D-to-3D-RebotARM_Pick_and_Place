"""Composite synthetic object views over a real RGB background (Phase 1B+).

Idea: instead of objects floating on black, paste the already-rendered synthetic
object (synth_render_<backend>/<capture>/*_view_*.png + .instance.png) on top of the
real empty-scene photo. Only the OBJECT is rendered from rotated virtual cameras
(yaw/pitch live in the synth render); the background is used flat, as captured. This
is a copy-paste style augmentation: no background reprojection, so no black
borders, and every view is usable.

Background source per capture (first match wins):
  1. --bg-capture override (one background reused for all captures)
  2. rgbd_backgrounds/<capture_id>/   (dedicated empty scene, recommended)
  3. --self-bg fallback: rgbd/<capture_id>/ (the original full scene; useful for
     a no-hardware smoke test, but the real object stays in the frame)

All images stay in BGR (OpenCV) end to end; no channel flips.

Usage:
  python DS/prelabel/composite_backgrounds.py --self-bg            # all captures
  python DS/prelabel/composite_backgrounds.py --capture capture_000004
  python DS/prelabel/composite_backgrounds.py --bg-capture capture_000002
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DS_DIR = SCRIPT_DIR.parent
REPO_ROOT = DS_DIR.parent
GEOMETRY_SRC = REPO_ROOT / "TabletopSeg3D" / "3DDetection" / "src"
for extra_path in (SCRIPT_DIR, GEOMETRY_SRC):
    if str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))

from labeled_gaussians import load_gaussians_meta  # noqa: E402
from lift_rgbd import (  # noqa: E402
    LABEL_JSON_NAMES,
    OCCLUDER_PREFIX,
    load_intrinsics,
    polygon_to_mask,
)

DEFAULT_RGBD_ROOT = DS_DIR / "dataset_capture" / "rgbd"
DEFAULT_BG_ROOT = DS_DIR / "dataset_capture" / "rgbd_backgrounds"
DEFAULT_SYNTH_ROOT = DS_DIR / "dataset_prelabel" / "synth_render_point"
DEFAULT_OUT_ROOT = DS_DIR / "dataset_prelabel" / "composited_views_point"


@dataclass(frozen=True)
class Background:
    background_id: str
    rgb_bgr: np.ndarray  # empty-scene photo (H, W, 3) BGR, used flat (no reprojection)
    intrinsics: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Composite synthetic object views over a real RGB background."
    )
    parser.add_argument("--synth-root", type=Path, default=DEFAULT_SYNTH_ROOT)
    parser.add_argument("--rgbd-root", type=Path, default=DEFAULT_RGBD_ROOT)
    parser.add_argument("--bg-root", type=Path, default=DEFAULT_BG_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--capture", default="", help="Single capture id (default: all in synth-root).")
    parser.add_argument(
        "--bg-capture",
        default="",
        help="Reuse one background (id under bg-root) for every capture instead of per-capture pairing.",
    )
    parser.add_argument(
        "--self-bg",
        action="store_true",
        help="Fallback: use rgbd/<capture_id> as its own background (no-hardware smoke test).",
    )
    return parser.parse_args()


def load_background(bg_dir: Path) -> Background:
    """Load the empty-scene photo (and intrinsics for metadata). Depth is not used:
    the background is composited flat, so there is no lift/reprojection step."""
    rgb_path = bg_dir / "rgb.png"
    intrinsics_path = bg_dir / "intrinsics.yaml"
    for required in (rgb_path, intrinsics_path):
        if not required.is_file():
            raise FileNotFoundError(f"{bg_dir.name}: missing {required.name}")

    rgb = cv2.imread(str(rgb_path))  # BGR
    if rgb is None:
        raise RuntimeError(f"{bg_dir.name}: cannot read {rgb_path}")
    intrinsics = load_intrinsics(intrinsics_path)
    return Background(bg_dir.name, rgb, intrinsics)


def background_frame(bg: Background, width: int, height: int) -> np.ndarray:
    """The flat background image, resized to the object view size if needed."""
    img = bg.rgb_bgr
    if img.shape[:2] != (height, width):
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    return img.copy()


def load_occluder_mask(rgbd_capture_dir: Path) -> np.ndarray | None:
    """Rasterize the occluder polygons (label prefix '_', e.g. the gripper) drawn
    on the capture's rgb.json. Returns a native-resolution bool mask, or None.

    The gripper sits in the real foreground; pasting a synthetic object over those
    pixels looks wrong, so we keep the background (gripper) on top there.
    """
    label_path = next(
        (rgbd_capture_dir / name for name in LABEL_JSON_NAMES if (rgbd_capture_dir / name).is_file()),
        None,
    )
    if label_path is None:
        return None

    data = json.loads(label_path.read_text(encoding="utf-8"))
    height = int(data.get("imageHeight") or 0)
    width = int(data.get("imageWidth") or 0)
    if not (height and width):
        rgb = cv2.imread(str(rgbd_capture_dir / "rgb.png"))
        if rgb is None:
            return None
        height, width = rgb.shape[:2]

    mask = np.zeros((height, width), dtype=bool)
    found = False
    for shape in data.get("shapes", []):
        label = (shape.get("label") or "").strip()
        points = shape.get("points") or []
        if not label.startswith(OCCLUDER_PREFIX) or len(points) < 3:
            continue
        mask |= polygon_to_mask(points, height, width)
        found = True
    return mask if found else None


def occluder_for_view(occluder_native: np.ndarray | None, width: int, height: int) -> np.ndarray:
    """Occluder mask resized to a view's resolution (empty if none)."""
    if occluder_native is None:
        return np.zeros((height, width), dtype=bool)
    if occluder_native.shape[:2] == (height, width):
        return occluder_native
    resized = cv2.resize(occluder_native.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def composite_object_over_background(
    bg_bgr: np.ndarray,
    obj_bgr: np.ndarray,
    instance_map: np.ndarray,
    occluder_mask: np.ndarray,
) -> np.ndarray:
    # Paste the object everywhere it exists EXCEPT where the foreground occluder
    # (gripper) is: there the real background stays on top.
    paste_mask = (instance_map > 0) & ~occluder_mask
    out = bg_bgr.copy()
    out[paste_mask] = obj_bgr[paste_mask]
    return out


def read_view_meta(meta_path: Path) -> dict[str, Any]:
    return yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}


def resolve_background_dir(
    capture_id: str, args: argparse.Namespace
) -> tuple[Path | None, str]:
    """Return (bg_dir, source_label). bg_dir None means no background available."""
    if args.bg_capture.strip():
        return args.bg_root.resolve() / args.bg_capture.strip(), "bg-capture"
    dedicated = args.bg_root.resolve() / capture_id
    if dedicated.is_dir():
        return dedicated, "rgbd_backgrounds"
    if args.self_bg:
        return args.rgbd_root.resolve() / capture_id, "self-bg"
    return None, "missing"


def clean_capture_output(out_dir: Path) -> None:
    if out_dir.is_dir():
        shutil.rmtree(out_dir)


def composite_capture(capture_dir: Path, args: argparse.Namespace) -> int:
    """Composite all synthetic views of one capture. Returns number of views written."""
    capture_id = capture_dir.name
    view_metas = sorted(capture_dir.glob(f"{capture_id}_view_*.meta.yaml"))
    if not view_metas:
        print(f"  SKIP {capture_id}: no synthetic views (run synth-render first)")
        return 0

    bg_dir, source = resolve_background_dir(capture_id, args)
    if bg_dir is None or not bg_dir.is_dir():
        print(
            f"  SKIP {capture_id}: no background. Capture paired bg with "
            f"`just capture --rgbd` (<- on presenter) or pass --self-bg / --bg-capture."
        )
        return 0

    background = load_background(bg_dir)
    capture_meta = load_gaussians_meta(capture_dir / "gaussians.npz")
    instances = capture_meta.get("instances") or []

    # Occluder (gripper) labeled on the capture's own rgb.json. The arm pose is the
    # same in the object and background captures, so a single native mask applies.
    occluder_native = load_occluder_mask(args.rgbd_root.resolve() / capture_id)

    # Render everything to memory first; only touch disk if the whole capture succeeds.
    rendered: list[tuple[str, np.ndarray, np.ndarray, dict[str, Any]]] = []
    dropped = 0
    for meta_path in view_metas:
        meta = read_view_meta(meta_path)
        view_id = meta.get("view_id") or meta_path.stem.split("_view_")[-1]
        width = int(meta["width"])
        height = int(meta["height"])
        yaw = float(meta.get("yaw_deg", 0.0))
        pitch = float(meta.get("pitch_deg", 0.0))

        stem = f"{capture_id}_{view_id}"
        obj_rgb_path = capture_dir / f"{stem}.png"
        obj_inst_path = capture_dir / f"{stem}.instance.png"
        obj_bgr = cv2.imread(str(obj_rgb_path), cv2.IMREAD_COLOR)
        instance_map = cv2.imread(str(obj_inst_path), cv2.IMREAD_UNCHANGED)
        if obj_bgr is None or instance_map is None:
            print(f"  SKIP {stem}: missing object rgb/instance")
            continue

        # Drop the gripper pixels from the label too: where the occluder hides the
        # object, the visible pixels are background, so the mask must not claim them.
        occluder_mask = occluder_for_view(occluder_native, width, height)
        instance_map[occluder_mask] = 0

        # Drop the view if no object is visible: either the rotation pushed it out
        # of frame, or the occluder hides it entirely (an unlabeled background frame).
        if not np.any(instance_map > 0):
            dropped += 1
            print(f"  DROP {stem}: no visible object (out of frame or fully occluded)")
            continue

        bg_bgr = background_frame(background, width, height)
        composited = composite_object_over_background(bg_bgr, obj_bgr, instance_map, occluder_mask)

        out_meta = {
            "capture_id": capture_id,
            "background_id": background.background_id,
            "background_source": source,
            "view_id": view_id,
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "backend": meta.get("backend", "points"),
            "width": width,
            "height": height,
            "occluder_applied": bool(occluder_native is not None),
            "intrinsics": background.intrinsics,
            "instances": instances,
        }
        rendered.append((view_id, composited, instance_map.astype(np.uint16), out_meta))

    # Always regenerate the capture's output folder so stale views never linger.
    out_capture_dir = args.output_root.resolve() / capture_id
    clean_capture_output(out_capture_dir)
    if not rendered:
        print(f"  {capture_id}: 0 views kept ({dropped} dropped) [bg={background.background_id}/{source}]")
        return 0

    out_capture_dir.mkdir(parents=True, exist_ok=True)
    for view_id, composited, instance_map, out_meta in rendered:
        view_dir = out_capture_dir / view_id
        view_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(view_dir / "rgb.png"), composited)
        cv2.imwrite(str(view_dir / "instance.png"), instance_map)
        (view_dir / "metadata.json").write_text(
            json.dumps(out_meta, indent=2), encoding="utf-8"
        )
    drop_hint = f", {dropped} dropped" if dropped else ""
    print(
        f"  {capture_id}: {len(rendered)} composited view(s){drop_hint} "
        f"[bg={background.background_id}/{source}]"
    )
    return len(rendered)


def list_synth_captures(synth_root: Path, capture_filter: str) -> list[Path]:
    synth_root = synth_root.resolve()
    if not synth_root.is_dir():
        raise SystemExit(f"Not found: {synth_root} (run just synth-render --backend ... first)")
    dirs = sorted(p for p in synth_root.iterdir() if p.is_dir() and p.name.startswith("capture_"))
    if capture_filter:
        dirs = [p for p in dirs if p.name == capture_filter]
        if not dirs:
            raise SystemExit(f"Capture not found under synth-root: {capture_filter}")
    return dirs


def main() -> None:
    args = parse_args()
    captures = list_synth_captures(args.synth_root, args.capture.strip())
    print(f"Compositing {len(captures)} capture(s) -> {args.output_root.resolve()}")

    total_views = 0
    ok = 0
    for capture_dir in captures:
        try:
            n = composite_capture(capture_dir, args)
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
            print(f"  SKIP {capture_dir.name}: {exc}")
            continue
        if n > 0:
            ok += 1
            total_views += n

    print("\n=== Composite summary ===")
    print(f"Captures composited: {ok}/{len(captures)}")
    print(f"Total composited views: {total_views}")
    if total_views == 0:
        raise SystemExit(
            "No views composited. Capture paired backgrounds "
            "(just capture --rgbd, <-) or run with --self-bg."
        )
    print("\nNext: just export-composite")


if __name__ == "__main__":
    main()
