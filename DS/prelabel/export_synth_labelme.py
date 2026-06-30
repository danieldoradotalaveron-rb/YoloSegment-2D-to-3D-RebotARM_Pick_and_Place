"""Export synthetic views (RGB + instance_id mask) to LabelMe JSON for review.

Reads build artifacts under dataset_prelabel/synth_render_<backend>/:
  capture_XXX_view_YYY.png
  capture_XXX_view_YYY.instance.png
  gaussians.meta.yaml  (instance_id -> class_name)

Writes human-review pairs to dataset_prelabel/to_review_synth_<backend>/.

Safe to re-run: existing review JSON is preserved unless --overwrite.

Usage:
  python DS/prelabel/export_synth_labelme.py
  python DS/prelabel/export_synth_labelme.py --capture capture_000004
  python DS/prelabel/export_synth_labelme.py --overwrite
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
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from labeled_gaussians import load_gaussians_meta

DS_DIR = SCRIPT_DIR.parent
DEFAULT_SYNTH_ROOT = DS_DIR / "dataset_prelabel" / "synth_render_point"
DEFAULT_OUTPUT = DS_DIR / "dataset_prelabel" / "to_review_synth_point"
LABELME_VERSION = "5.5.0"


@dataclass(frozen=True)
class SynthView:
    rgb_path: Path
    instance_path: Path
    capture_dir: Path
    stem: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export synth_render_<backend> views to LabelMe JSON (to_review_synth_<backend>/). "
        "Default: all views under synth-root."
    )
    parser.add_argument("--synth-root", type=Path, default=DEFAULT_SYNTH_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--capture", default="", help="Single capture id (default: all).")
    parser.add_argument("--min-area", type=int, default=50, help="Drop contours smaller than this (px).")
    parser.add_argument(
        "--morph-kernel",
        type=int,
        default=5,
        help="Morphological close kernel (px) to connect splat fragments (0=off).",
    )
    parser.add_argument(
        "--epsilon-ratio",
        type=float,
        default=0.005,
        help="Polygon simplification as fraction of contour perimeter.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-export even if review JSON already exists (destroys manual edits).",
    )
    return parser.parse_args()


def instance_id_to_class(capture_meta: dict[str, Any]) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for item in capture_meta.get("instances") or []:
        inst_id = int(item["instance_id"])
        mapping[inst_id] = str(item["class_name"])
    return mapping


def iter_synth_views(synth_root: Path, capture_filter: str) -> list[SynthView]:
    synth_root = synth_root.resolve()
    if not synth_root.is_dir():
        raise SystemExit(f"Not found: {synth_root} (run just synth-render --backend ... first)")

    views: list[SynthView] = []
    capture_dirs = sorted(p for p in synth_root.iterdir() if p.is_dir() and p.name.startswith("capture_"))
    if capture_filter:
        capture_dirs = [p for p in capture_dirs if p.name == capture_filter]
        if not capture_dirs:
            raise SystemExit(f"Capture not found under synth-root: {capture_filter}")

    for capture_dir in capture_dirs:
        prefix = capture_dir.name
        for rgb_path in sorted(capture_dir.glob(f"{prefix}_view_*.png")):
            if rgb_path.name.endswith(".instance.png"):
                continue
            stem = rgb_path.stem
            instance_path = capture_dir / f"{stem}.instance.png"
            if not instance_path.is_file():
                print(f"  skip (no instance mask): {rgb_path.name}")
                continue
            views.append(
                SynthView(
                    rgb_path=rgb_path,
                    instance_path=instance_path,
                    capture_dir=capture_dir,
                    stem=stem,
                )
            )
    return views


def mask_to_polygons(
    binary: np.ndarray,
    *,
    min_area: int,
    epsilon_ratio: float,
) -> list[list[list[float]]]:
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[list[float]]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        epsilon = max(1.0, epsilon_ratio * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        points = [[float(x), float(y)] for x, y in approx.reshape(-1, 2)]
        polygons.append(points)
    return polygons


def instance_map_to_shapes(
    inst_map: np.ndarray,
    id_to_class: dict[int, str],
    *,
    min_area: int,
    morph_kernel: int,
    epsilon_ratio: float,
) -> list[dict[str, Any]]:
    shapes: list[dict[str, Any]] = []
    instance_ids = sorted(i for i in np.unique(inst_map) if int(i) > 0)

    for inst_id in instance_ids:
        label = id_to_class.get(int(inst_id))
        if not label:
            print(f"    warn: instance_id {inst_id} not in gaussians.meta.yaml — skipped")
            continue

        binary = np.where(inst_map == inst_id, 255, 0).astype(np.uint8)
        if morph_kernel > 0:
            k = max(1, morph_kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        for polygon in mask_to_polygons(binary, min_area=min_area, epsilon_ratio=epsilon_ratio):
            shapes.append(
                {
                    "label": label,
                    "points": polygon,
                    "group_id": int(inst_id),
                    "shape_type": "polygon",
                    "flags": {},
                }
            )
    return shapes


def build_labelme(
    shapes: list[dict[str, Any]],
    image_name: str,
    width: int,
    height: int,
) -> dict[str, Any]:
    return {
        "version": LABELME_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }


def export_view(
    view: SynthView,
    output_dir: Path,
    *,
    min_area: int,
    morph_kernel: int,
    epsilon_ratio: float,
    overwrite: bool,
) -> tuple[str, int]:
    """Return (status, shape_count): exported | kept | skipped."""
    out_json = output_dir / f"{view.stem}.json"
    out_image = output_dir / view.rgb_path.name

    if out_json.exists() and not overwrite:
        return "kept", 0

    meta_path = view.capture_dir / "gaussians.npz"
    capture_meta = load_gaussians_meta(meta_path)
    if not capture_meta:
        raise RuntimeError(f"{view.capture_dir.name}: missing gaussians.meta.yaml")

    id_to_class = instance_id_to_class(capture_meta)
    inst_map = cv2.imread(str(view.instance_path), cv2.IMREAD_UNCHANGED)
    if inst_map is None:
        raise RuntimeError(f"failed to read {view.instance_path}")

    rgb = cv2.imread(str(view.rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"failed to read {view.rgb_path}")

    height, width = inst_map.shape[:2]
    shapes = instance_map_to_shapes(
        inst_map,
        id_to_class,
        min_area=min_area,
        morph_kernel=morph_kernel,
        epsilon_ratio=epsilon_ratio,
    )
    if not shapes:
        return "skipped", 0

    labelme = build_labelme(shapes, view.rgb_path.name, width, height)
    shutil.copy2(view.rgb_path, out_image)
    out_json.write_text(json.dumps(labelme, indent=2), encoding="utf-8")
    return "exported", len(shapes)


def main() -> None:
    args = parse_args()
    synth_root = args.synth_root.resolve()
    output_dir = args.output.resolve()
    views = iter_synth_views(synth_root, args.capture.strip())
    if not views:
        raise SystemExit(f"No synthetic views found under {synth_root}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("export_synth_labelme")
    print(f"  synth-root : {synth_root}")
    print(f"  output     : {output_dir}")
    print(f"  views      : {len(views)}")
    print(f"  min-area   : {args.min_area}")
    print()

    exported = kept = skipped = failed = 0
    total_shapes = 0

    for view in views:
        try:
            status, n_shapes = export_view(
                view,
                output_dir,
                min_area=args.min_area,
                morph_kernel=args.morph_kernel,
                epsilon_ratio=args.epsilon_ratio,
                overwrite=args.overwrite,
            )
        except (RuntimeError, OSError, ValueError) as exc:
            failed += 1
            print(f"  FAIL {view.stem}: {exc}")
            continue

        if status == "exported":
            exported += 1
            total_shapes += n_shapes
            print(f"  export: {view.stem}.json ({n_shapes} shapes)")
        elif status == "kept":
            kept += 1
            print(f"  keep (already in review): {view.stem}.json")
        else:
            skipped += 1
            print(f"  skip (no shapes): {view.stem}")

    print(f"\n=== Export synth summary ===")
    print(f"Exported: {exported}  shapes: {total_shapes}")
    if kept:
        print(f"Kept (existing review): {kept}  (use --overwrite to replace)")
    if skipped:
        print(f"Skipped (empty mask): {skipped}")
    if failed:
        print(f"Failed: {failed}")
    if exported == 0 and kept == 0:
        raise SystemExit("Nothing exported.")

    print(f"\nNext: just review-synth  (QC only; object-on-black is NOT promoted)")
    print(f"Then: just composite-synth  -> composited_views_<backend>/")


if __name__ == "__main__":
    main()
