"""
Export prelabels (YOLO or SAM3) to LabelMe JSON for human review.

Offline tooling only. This is NOT part of the runtime (TabletopSeg3D).

How it works:
  Reads the prelabel JSON written by prelabel_images.py (prelabels_yolo/ or
  prelabels_sam3/), copies each source image and writes a LabelMe-compatible .json
  next to it in the matching to_review_<backend>/. A human then opens that folder in LabelMe (or
  imports it into CVAT) to fix polygons/labels. After review, promote it into
  DS/dataset_labeled/ and convert to YOLO-Seg with DS/convert_labelme_to_yolo.py.

Safe to re-run: existing review files are PRESERVED (never clobbered). Pass
--overwrite only if you want to discard manual review and regenerate.

Usage:
  python export_to_labelme.py
  python export_to_labelme.py --min-score 0.4
  python export_to_labelme.py --overwrite   # discards existing review
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DS_DIR = SCRIPT_DIR.parent
DATASET_ROOT = DS_DIR / "dataset_prelabel"

# Direct-run defaults target the YOLO folders; the `just` recipes pass explicit
# --prelabels/--output per backend (yolo -> *_yolo, sam3 -> *_sam3).
DEFAULT_PRELABELS = DATASET_ROOT / "prelabels_yolo"
DEFAULT_IMAGES = DATASET_ROOT / "input_images"
DEFAULT_OUTPUT = DATASET_ROOT / "to_review_yolo"

LABELME_VERSION = "5.5.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export prelabels (YOLO or SAM3) to LabelMe JSON for human review.",
    )
    parser.add_argument("--prelabels", type=Path, default=DEFAULT_PRELABELS, help="Folder with prelabel JSON.")
    parser.add_argument("--images", type=Path, default=DEFAULT_IMAGES, help="Folder with the source images.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output folder (LabelMe .jpg + .json).")
    parser.add_argument("--min-score", type=float, default=0.0, help="Drop detections below this score.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-export even if a reviewed .json already exists (DESTROYS manual review). Off by default.",
    )
    return parser.parse_args()


def to_labelme(record: dict, image_name: str, min_score: float) -> dict:
    shapes = []
    for det in record.get("detections", []):
        if det.get("score", 1.0) < min_score:
            continue
        polygon = det.get("polygon", [])
        if len(polygon) < 3:
            continue
        shapes.append(
            {
                "label": det["class"],
                "points": [[float(x), float(y)] for x, y in polygon],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {},
            }
        )
    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": record["height"],
        "imageWidth": record["width"],
    }


def main() -> None:
    args = parse_args()
    prelabels_dir = args.prelabels.resolve()
    images_dir = args.images.resolve()
    output_dir = args.output.resolve()

    if not prelabels_dir.exists():
        raise SystemExit(
            f"ERROR: prelabels folder not found: {prelabels_dir}\n       Run 'just prelabel-yolo' (or 'just prelabel-sam3') first."
        )
    if not images_dir.exists():
        raise SystemExit(f"ERROR: images folder not found: {images_dir}")

    json_files = sorted(prelabels_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"ERROR: no prelabel JSON in {prelabels_dir}. Run 'just prelabel-yolo' (or 'just prelabel-sam3') first.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("export_to_labelme")
    print(f"  prelabels : {prelabels_dir}")
    print(f"  images    : {images_dir}")
    print(f"  output dir: {output_dir}")
    print(f"  min score : {args.min_score}")
    print()

    exported = 0
    total_shapes = 0
    skipped = 0
    preserved = 0
    for json_path in json_files:
        record = json.loads(json_path.read_text(encoding="utf-8"))
        image_name = record.get("image", f"{json_path.stem}.jpg")
        src_image = images_dir / image_name
        if not src_image.exists():
            skipped += 1
            print(f"  skip (no image): {image_name}")
            continue

        out_json = output_dir / f"{src_image.stem}.json"
        # Non-destructive by default: never clobber a review the human may have edited.
        if out_json.exists() and not args.overwrite:
            preserved += 1
            print(f"  keep (already in review): {out_json.name}")
            continue

        labelme = to_labelme(record, image_name, args.min_score)
        shutil.copy2(src_image, output_dir / image_name)
        out_json.write_text(json.dumps(labelme, indent=2), encoding="utf-8")
        exported += 1
        total_shapes += len(labelme["shapes"])

    print(f"\nDone. {exported} images, {total_shapes} shapes -> {output_dir}")
    if preserved:
        print(f"Preserved (existing review kept; use --overwrite to replace): {preserved}")
    if skipped:
        print(f"Skipped (prelabel without image): {skipped}")
    print()
    print("Review in LabelMe / CVAT, then convert to YOLO-Seg with:")
    print(f"  python {DS_DIR / 'convert_labelme_to_yolo.py'} \\")
    print(f"    --input {output_dir} \\")
    print(f"    --output {DATASET_ROOT / 'exports_yolo'}")


if __name__ == "__main__":
    main()
