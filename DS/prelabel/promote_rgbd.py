"""Scatter reviewed RGBD prelabels back into the capture folders for the synth step.

Offline tooling only. Reads dataset_prelabel/to_review_rgbd_yolo/<capture_id>.json
(reviewed in LabelMe) and writes it as rgb.json inside the matching
dataset_capture/rgbd/<capture_id>/ -- the exact input lift_rgbd.py / synth-render read.

Idempotent and non-destructive:
  - imagePath is rewritten to rgb.png (the name inside the capture folder).
  - never clobbers an rgb.json that differs from what we would write (you may have
    edited it by hand in the capture folder); identical content is a harmless no-op.

Usage:
  python promote_rgbd.py
  python promote_rgbd.py --capture capture_000004
  python promote_rgbd.py --overwrite      # replace differing rgb.json too
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DS_DIR = SCRIPT_DIR.parent
DEFAULT_REVIEW_ROOT = DS_DIR / "dataset_prelabel" / "to_review_rgbd_yolo"
DEFAULT_RGBD_ROOT = DS_DIR / "dataset_capture" / "rgbd"
RGB_NAME = "rgb.png"
LABEL_NAME = "rgb.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Promote reviewed to_review_rgbd_yolo/ labels into rgbd/<id>/rgb.json."
    )
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--rgbd-root", type=Path, default=DEFAULT_RGBD_ROOT)
    parser.add_argument("--capture", default="", help="Single capture id (default: all reviewed).")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an rgb.json that differs from the reviewed label (default: keep + warn).",
    )
    return parser.parse_args()


def normalized_label(review_json: Path) -> str:
    """LabelMe content rewritten to reference rgb.png, serialized deterministically."""
    data = json.loads(review_json.read_text(encoding="utf-8"))
    data["imagePath"] = RGB_NAME
    data["imageData"] = None
    return json.dumps(data, indent=2)


def main() -> None:
    args = parse_args()
    review_root = args.review_root.resolve()
    rgbd_root = args.rgbd_root.resolve()
    if not review_root.is_dir():
        raise SystemExit(f"Not found: {review_root} (run 'just prelabel-rgbd' and review first)")

    capture_filter = args.capture.strip()
    review_jsons = sorted(review_root.glob("*.json"))
    if capture_filter:
        review_jsons = [p for p in review_jsons if p.stem == capture_filter]
    if not review_jsons:
        raise SystemExit(f"No reviewed .json in {review_root} (filter: {capture_filter or 'none'})")

    promoted = 0
    kept = 0
    missing = 0
    for review_json in review_jsons:
        capture_id = review_json.stem
        capture_dir = rgbd_root / capture_id
        if not (capture_dir / RGB_NAME).is_file():
            missing += 1
            print(f"  skip (no capture {RGB_NAME}): {capture_id}")
            continue

        content = normalized_label(review_json)
        dst = capture_dir / LABEL_NAME
        if dst.is_file() and not args.overwrite:
            if dst.read_text(encoding="utf-8") == content:
                continue  # identical -> idempotent no-op
            kept += 1
            print(f"  keep (capture rgb.json differs, NOT overwriting): {capture_id}")
            continue

        dst.write_text(content, encoding="utf-8")
        promoted += 1
        print(f"  promoted: {review_json.name} -> {dst}")

    print(f"\nPromoted {promoted} label(s) into {rgbd_root}/<id>/{LABEL_NAME}")
    if kept:
        print(f"Kept {kept} differing capture label(s) (edit review + --overwrite to update).")
    if missing:
        print(f"Skipped {missing} review(s) without a matching capture folder.")
    print("Next: just synth-render --backend points --inpaint-depth  (then composite-synth / export / review / promote)")


if __name__ == "__main__":
    main()
