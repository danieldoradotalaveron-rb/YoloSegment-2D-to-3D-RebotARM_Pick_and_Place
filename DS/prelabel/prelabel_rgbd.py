"""YOLO pre-label the RGBD captures so the synth pipeline gets masks to lift.

Offline tooling only (NOT the runtime). This is the auto-label counterpart of the
manual "draw rgb.json in LabelMe" step: it runs the trained YOLO-seg model on every
dataset_capture/rgbd/capture_*/rgb.png and writes a LabelMe pair (renamed by
capture id) into dataset_prelabel/to_review_rgbd_yolo/ for human review.

After review, `promote_rgbd.py` scatters each <capture_id>.json back as rgb.json
inside the matching rgbd/capture_*/ folder (the input the synth pipeline reads).

Why a dedicated script (vs prelabel-yolo on input_images/):
  - source images live in per-capture subfolders, all named rgb.png -> must be
    renamed by capture id to share a single LabelMe review folder.
  - lower default conf: the synth pipeline lifts EVERY mask to 3D, so a missed
    instance hurts more than a false positive you can delete in review.
  - skips captures that already have a label, so re-runs only touch new captures.

Usage:
  python prelabel_rgbd.py
  python prelabel_rgbd.py --capture capture_000004
  python prelabel_rgbd.py --conf 0.3 --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_to_labelme import to_labelme  # noqa: E402
from prelabel_images import YoloBackend  # noqa: E402

DS_DIR = SCRIPT_DIR.parent
DEFAULT_RGBD_ROOT = DS_DIR / "dataset_capture" / "rgbd"
DEFAULT_OUTPUT = DS_DIR / "dataset_prelabel" / "to_review_rgbd_yolo"
RGB_NAME = "rgb.png"
# Same label names the manual flow / lift_rgbd look for next to the capture.
LABEL_JSON_NAMES = ("rgb.json", "labelme.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO pre-label rgbd/ captures into to_review_rgbd_yolo/ for review."
    )
    parser.add_argument("--rgbd-root", type=Path, default=DEFAULT_RGBD_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--capture", default="", help="Single capture id (default: all).")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="YOLO weights (default: latest runs/segment/*/weights/best.pt).",
    )
    # Lower than prelabel-yolo's 0.8: the synth pipeline needs every instance.
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold (default: 0.4).")
    parser.add_argument("--device", type=str, default="0", help="Inference device (e.g. 0, cpu).")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size (default: 1024).")
    parser.add_argument("--half", action="store_true", help="FP16 inference (GPU only).")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-predict captures that already have a label and replace existing reviews.",
    )
    return parser.parse_args()


def has_label(capture_dir: Path) -> bool:
    return any((capture_dir / name).is_file() for name in LABEL_JSON_NAMES)


def iter_capture_dirs(rgbd_root: Path, capture_filter: str) -> list[Path]:
    if not rgbd_root.is_dir():
        raise SystemExit(f"Not found: {rgbd_root}")
    if capture_filter:
        target = rgbd_root / capture_filter
        if not (target / RGB_NAME).is_file():
            raise SystemExit(f"Capture not found or missing {RGB_NAME}: {target}")
        return [target]
    dirs = sorted(
        p
        for p in rgbd_root.iterdir()
        if p.is_dir() and p.name.startswith("capture_") and (p / RGB_NAME).is_file()
    )
    if not dirs:
        raise SystemExit(f"No capture_*/{RGB_NAME} under {rgbd_root}")
    return dirs


def main() -> None:
    args = parse_args()
    rgbd_root = args.rgbd_root.resolve()
    output_dir = args.output.resolve()
    capture_dirs = iter_capture_dirs(rgbd_root, args.capture.strip())

    backend = YoloBackend(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("prelabel_rgbd (backend=yolo)")
    print(f"  rgbd root : {rgbd_root}")
    print(f"  output dir: {output_dir}")
    print(f"  model     : {backend.model_path}")
    print(backend.describe())
    print(f"  conf      : {args.conf} | device: {args.device}")
    print(f"  captures  : {len(capture_dirs)}")
    print()

    exported = 0
    skipped_labeled = 0
    preserved = 0
    total_shapes = 0
    for capture_dir in capture_dirs:
        capture_id = capture_dir.name
        # Skip captures you already labeled (manually or promoted) unless forced:
        # re-runs then only touch new captures (idempotent).
        if has_label(capture_dir) and not args.overwrite:
            skipped_labeled += 1
            print(f"  skip (already labeled): {capture_id}")
            continue

        out_json = output_dir / f"{capture_id}.json"
        if out_json.exists() and not args.overwrite:
            preserved += 1
            print(f"  keep (already in review): {out_json.name}")
            continue

        src_image = capture_dir / RGB_NAME
        out_image_name = f"{capture_id}.png"
        record = backend.infer(src_image)
        labelme = to_labelme(record, out_image_name, min_score=0.0)
        shutil.copy2(src_image, output_dir / out_image_name)
        out_json.write_text(json.dumps(labelme, indent=2), encoding="utf-8")
        exported += 1
        total_shapes += len(labelme["shapes"])
        print(f"  [{capture_id}] {len(labelme['shapes'])} shapes")

    print(f"\nDone. {exported} prelabel(s), {total_shapes} shapes -> {output_dir}")
    if skipped_labeled:
        print(f"Skipped (already labeled; --overwrite to redo): {skipped_labeled}")
    if preserved:
        print(f"Preserved (existing review kept; --overwrite to replace): {preserved}")
    print("Next: just review-rgbd  (fix masks), then just promote-rgbd")


if __name__ == "__main__":
    main()
