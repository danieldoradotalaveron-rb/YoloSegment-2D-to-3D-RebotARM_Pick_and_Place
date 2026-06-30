"""Export composited views (RGB over real background + instance mask) to LabelMe.

Reads build artifacts under dataset_prelabel/composited_views_<backend>/:
  <capture>/<view>/rgb.png
  <capture>/<view>/instance.png
  <capture>/<view>/metadata.json   (carries instance_id -> class_name)

Writes flat human-review pairs to dataset_prelabel/to_review_composite_<backend>/:
  <capture>_<view>.png
  <capture>_<view>.json   (LabelMe polygons)

Shares the instance-mask -> polygon logic with export_synth_labelme so the two
exporters stay consistent. Safe to re-run: existing review JSON is preserved
unless --overwrite.

Usage:
  python DS/prelabel/export_composite_labelme.py
  python DS/prelabel/export_composite_labelme.py --capture capture_000004
  python DS/prelabel/export_composite_labelme.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_synth_labelme import (  # noqa: E402 - reuse shared mask->LabelMe logic
    build_labelme,
    instance_id_to_class,
    instance_map_to_shapes,
)

DS_DIR = SCRIPT_DIR.parent
DEFAULT_COMPOSITE_ROOT = DS_DIR / "dataset_prelabel" / "composited_views_point"
DEFAULT_OUTPUT = DS_DIR / "dataset_prelabel" / "to_review_composite_point"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export composited_views_<backend> to LabelMe JSON (to_review_composite_<backend>/)."
    )
    parser.add_argument("--composite-root", type=Path, default=DEFAULT_COMPOSITE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--capture", default="", help="Single capture id (default: all).")
    parser.add_argument("--min-area", type=int, default=50, help="Drop contours smaller than this (px).")
    parser.add_argument("--morph-kernel", type=int, default=5, help="Close kernel to connect fragments (0=off).")
    parser.add_argument("--epsilon-ratio", type=float, default=0.005, help="Polygon simplification ratio.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-export even if review JSON already exists (destroys manual edits).",
    )
    return parser.parse_args()


def iter_view_dirs(composite_root: Path, capture_filter: str) -> list[Path]:
    composite_root = composite_root.resolve()
    if not composite_root.is_dir():
        raise SystemExit(f"Not found: {composite_root} (run just composite-synth first)")
    capture_dirs = sorted(
        p for p in composite_root.iterdir() if p.is_dir() and p.name.startswith("capture_")
    )
    if capture_filter:
        capture_dirs = [p for p in capture_dirs if p.name == capture_filter]
        if not capture_dirs:
            raise SystemExit(f"Capture not found under composite-root: {capture_filter}")

    view_dirs: list[Path] = []
    for capture_dir in capture_dirs:
        for view_dir in sorted(p for p in capture_dir.iterdir() if p.is_dir()):
            if (view_dir / "rgb.png").is_file() and (view_dir / "instance.png").is_file():
                view_dirs.append(view_dir)
    return view_dirs


def load_metadata(view_dir: Path) -> dict[str, Any]:
    meta_path = view_dir / "metadata.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def export_view(view_dir: Path, output_dir: Path, args: argparse.Namespace) -> tuple[str, int]:
    capture_id = view_dir.parent.name
    view_id = view_dir.name
    stem = f"{capture_id}_{view_id}"
    out_json = output_dir / f"{stem}.json"

    if out_json.exists() and not args.overwrite:
        return "kept", 0

    meta = load_metadata(view_dir)
    id_to_class = instance_id_to_class(meta)
    if not id_to_class:
        raise RuntimeError(f"{stem}: metadata.json missing instances map")

    inst_map = cv2.imread(str(view_dir / "instance.png"), cv2.IMREAD_UNCHANGED)
    if inst_map is None:
        raise RuntimeError(f"{stem}: cannot read instance.png")

    height, width = inst_map.shape[:2]
    shapes = instance_map_to_shapes(
        inst_map,
        id_to_class,
        min_area=args.min_area,
        morph_kernel=args.morph_kernel,
        epsilon_ratio=args.epsilon_ratio,
    )
    if not shapes:
        return "skipped", 0

    image_name = f"{stem}.png"
    labelme = build_labelme(shapes, image_name, width, height)
    shutil.copy2(view_dir / "rgb.png", output_dir / image_name)
    out_json.write_text(json.dumps(labelme, indent=2), encoding="utf-8")
    return "exported", len(shapes)


def main() -> None:
    args = parse_args()
    output_dir = args.output.resolve()
    view_dirs = iter_view_dirs(args.composite_root, args.capture.strip())
    if not view_dirs:
        raise SystemExit(f"No composited views under {args.composite_root.resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print("export_composite_labelme")
    print(f"  composite-root: {args.composite_root.resolve()}")
    print(f"  output        : {output_dir}")
    print(f"  views         : {len(view_dirs)}")
    print()

    exported = kept = skipped = failed = 0
    total_shapes = 0
    for view_dir in view_dirs:
        try:
            status, n_shapes = export_view(view_dir, output_dir, args)
        except (RuntimeError, OSError, ValueError) as exc:
            failed += 1
            print(f"  FAIL {view_dir.parent.name}_{view_dir.name}: {exc}")
            continue
        stem = f"{view_dir.parent.name}_{view_dir.name}"
        if status == "exported":
            exported += 1
            total_shapes += n_shapes
            print(f"  export: {stem}.json ({n_shapes} shapes)")
        elif status == "kept":
            kept += 1
            print(f"  keep (already in review): {stem}.json")
        else:
            skipped += 1
            print(f"  skip (no shapes): {stem}")

    print(f"\n=== Export composite summary ===")
    print(f"Exported: {exported}  shapes: {total_shapes}")
    if kept:
        print(f"Kept (existing review): {kept}  (use --overwrite to replace)")
    if skipped:
        print(f"Skipped (empty mask): {skipped}")
    if failed:
        print(f"Failed: {failed}")
    if exported == 0 and kept == 0:
        raise SystemExit("Nothing exported.")

    print(f"\nNext: just review-composite")
    print(f"Then: just promote-composite  -> dataset_labeled/composited_labelme/")


if __name__ == "__main__":
    main()
