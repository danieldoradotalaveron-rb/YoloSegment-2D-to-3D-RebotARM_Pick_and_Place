"""
Pre-label NEW images and write predicted polygons (one JSON per image).

Offline tooling only. This is NOT part of the runtime (TabletopSeg3D).

Two backends (same output format, shared post-processing):

- yolo (default): use the project's trained YOLO-seg model
  (runs/segment/*/weights/best.pt). Best for the classes it already knows
  (whatever it was trained on) and needs no extra download.

- sam3: Ultralytics SAM3 text-based concept segmentation
  (`SAM3SemanticPredictor`). Useful for new/unknown classes. SAM3 weights
  (sam3.pt) are gated and NOT auto-downloaded; request access and download from
  https://huggingface.co/facebook/sam3 . Classes and their text prompts are
  defined in DS/prelabel/sam3_classes.yaml (the single source for the SAM3
  pipeline); edit that file to tune ambiguous concepts.

Output: <output>/<stem>.json with class, score, bbox and polygon for each
detected instance (the `just` recipes use prelabels_yolo/ or prelabels_sam3/).
Convert to LabelMe with export_to_labelme.py.

Usage:
  python prelabel_images.py                       # YOLO backend, latest best.pt
  python prelabel_images.py --model runs/segment/train/weights/best.pt
  python prelabel_images.py --backend sam3 --device 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DS_DIR = SCRIPT_DIR.parent
REPO_ROOT = DS_DIR.parent
DATASET_ROOT = DS_DIR / "dataset_prelabel"

DEFAULT_INPUT = DATASET_ROOT / "input_images"
# Direct-run default targets the YOLO folder (the default backend). The `just`
# recipes always pass an explicit --output per backend (prelabels_yolo / prelabels_sam3).
DEFAULT_OUTPUT = DATASET_ROOT / "prelabels_yolo"
RUNS_SEGMENT = REPO_ROOT / "runs" / "segment"
DEFAULT_SAM3_MODEL = REPO_ROOT / "sam3.pt"
# Single source of truth for SAM3 class names + text prompts (see the YAML for format).
DEFAULT_SAM3_CLASSES = SCRIPT_DIR / "sam3_classes.yaml"

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
HF_URL = "https://huggingface.co/facebook/sam3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-label new images with a YOLO-seg model (default) or SAM3.",
    )
    parser.add_argument("--backend", choices=["yolo", "sam3"], default="yolo", help="Model backend.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Folder with new, unlabeled images.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output folder for prelabel JSON.")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Weights path. YOLO: defaults to latest runs/segment/*/weights/best.pt. SAM3: defaults to ./sam3.pt.",
    )
    parser.add_argument("--conf", type=float, default=0.8, help="Confidence threshold.")
    parser.add_argument(
        "--dedup-iou",
        type=float,
        default=0.7,
        help=(
            "Merge same-class detections whose MASK IoU exceeds this, keeping the "
            "highest-score one (box NMS misses near-identical masks). 0 disables."
        ),
    )
    parser.add_argument("--device", type=str, default="0", help="Inference device (e.g. 0, cpu).")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size (default: 1024). Higher = finer masks/polygons.")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference (GPU only).")
    # SAM3-only: classes and text prompts come from a YAML config (single source of truth).
    parser.add_argument(
        "--classes-config",
        type=Path,
        default=DEFAULT_SAM3_CLASSES,
        help="[sam3] YAML with class names + text prompts (default: DS/prelabel/sam3_classes.yaml).",
    )
    return parser.parse_args()


def resolve_latest_best_pt() -> Path | None:
    if not RUNS_SEGMENT.is_dir():
        return None
    candidates = sorted(
        RUNS_SEGMENT.glob("*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_sam3_classes(config_path: Path) -> tuple[list[str], list[str]]:
    """Read class names and SAM3 text prompts from the YAML config.

    Returns (names, prompts) in declaration order. `prompt` defaults to `name`.
    This YAML is the single source of truth for the whole SAM3 pipeline.
    """
    if not config_path.is_file():
        raise SystemExit(
            f"ERROR: SAM3 classes config not found: {config_path}\n"
            "       Create it (see DS/prelabel/sam3_classes.yaml) or pass --classes-config."
        )
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    entries = data.get("classes")
    if not entries:
        raise SystemExit(f"ERROR: no 'classes' defined in {config_path}")

    names: list[str] = []
    prompts: list[str] = []
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or "name" not in entry:
            raise SystemExit(
                f"ERROR: class #{i} in {config_path} must be a mapping with a 'name' "
                "(and an optional 'prompt')."
            )
        name = str(entry["name"])
        names.append(name)
        prompts.append(str(entry.get("prompt", name)))
    return names, prompts


def _dedup_by_mask_iou(detections: list[dict], masks: np.ndarray, iou_thr: float) -> tuple[list[dict], int]:
    """Drop same-class detections that overlap a higher-score one above ``iou_thr``.

    Operates on the real per-instance masks (``result.masks.data``), since box NMS
    keeps two near-identical masks when their *boxes* fall below its IoU threshold.
    Each detection carries an ``_idx`` row into ``masks``. Returns (kept, removed_count)
    with the kept detections in their original order.
    """
    order = sorted(range(len(detections)), key=lambda k: detections[k]["score"], reverse=True)
    kept: list[int] = []
    removed = 0
    for k in order:
        mk = masks[detections[k]["_idx"]]
        is_dup = False
        for j in kept:
            if detections[j]["class"] != detections[k]["class"]:
                continue
            mj = masks[detections[j]["_idx"]]
            inter = int(np.logical_and(mk, mj).sum())
            if inter == 0:
                continue
            union = int(np.logical_or(mk, mj).sum())
            if union and inter / union >= iou_thr:
                is_dup = True
                break
        if is_dup:
            removed += 1
        else:
            kept.append(k)
    return [detections[k] for k in sorted(kept)], removed


def results_to_record(result, names: dict[int, str], image_path: Path, dedup_iou: float = 0.0) -> dict:
    """Convert an Ultralytics Results object into our prelabel record (shared by both backends)."""
    image = cv2.imread(str(image_path))
    height, width = image.shape[:2]

    detections: list[dict] = []
    if result.masks is not None:
        polygons = result.masks.xy
        cls_ids = result.boxes.cls.tolist()
        scores = result.boxes.conf.tolist()
        boxes = result.boxes.xyxy.tolist()
        for idx, (polygon, cls_id, score, box) in enumerate(zip(polygons, cls_ids, scores, boxes)):
            if len(polygon) < 3:
                continue
            name = names.get(int(cls_id))
            if name is None:
                continue
            detections.append(
                {
                    "class": name,
                    "score": round(float(score), 4),
                    "bbox": [round(float(v), 2) for v in box],
                    "polygon": [[round(float(x), 2), round(float(y), 2)] for x, y in polygon],
                    "_idx": idx,
                }
            )

        # Box NMS already ran inside predict(); this extra pass removes the
        # same-class near-duplicate *masks* it leaves behind (see _dedup_by_mask_iou).
        masks_data = getattr(result.masks, "data", None)
        if dedup_iou and dedup_iou > 0 and masks_data is not None and detections:
            try:
                masks_arr = masks_data.cpu().numpy().astype(bool)
            except AttributeError:
                masks_arr = np.asarray(masks_data).astype(bool)
            detections, removed = _dedup_by_mask_iou(detections, masks_arr, dedup_iou)
            if removed:
                print(f"    [dedup] {image_path.name}: removed {removed} duplicate mask(s) (mask-IoU>={dedup_iou})")

        for det in detections:
            det.pop("_idx", None)

    return {"image": image_path.name, "width": width, "height": height, "detections": detections}


class YoloBackend:
    """Pre-label using the project's trained YOLO-seg model."""

    def __init__(self, args: argparse.Namespace):
        from ultralytics import YOLO

        model_path = args.model.resolve() if args.model else resolve_latest_best_pt()
        if model_path is None or not Path(model_path).is_file():
            raise SystemExit(
                "ERROR: YOLO weights not found.\n"
                f"       Looked under {RUNS_SEGMENT}/*/weights/best.pt\n"
                "       Train first ('just train') or pass --model /path/to/best.pt"
            )
        self.model_path = Path(model_path)
        self.model = YOLO(str(self.model_path))
        self.names = {int(i): n for i, n in self.model.names.items()}
        # Infer above the native image size (640x480) so masks/polygons keep fine detail.
        self.imgsz = args.imgsz or 1024
        self.conf = args.conf
        self.device = args.device
        self.half = args.half
        self.dedup_iou = args.dedup_iou

    def describe(self) -> str:
        return f"  classes   : {', '.join(self.names[i] for i in sorted(self.names))}"

    def infer(self, image_path: Path) -> dict:
        results = self.model.predict(
            source=str(image_path),
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            half=self.half,
            retina_masks=True,
            verbose=False,
        )
        return results_to_record(results[0], self.names, image_path, self.dedup_iou)


class Sam3Backend:
    """Pre-label using Ultralytics SAM3 text-based concept segmentation."""

    def __init__(self, args: argparse.Namespace):
        from ultralytics.models.sam import SAM3SemanticPredictor

        model_path = (args.model or DEFAULT_SAM3_MODEL).resolve()
        if not model_path.is_file():
            raise SystemExit(
                f"ERROR: SAM3 weights not found: {model_path}\n"
                f"       SAM3 weights are gated and not auto-downloaded.\n"
                f"       Request access and download sam3.pt from {HF_URL}\n"
                f"       then place it at {model_path} or pass --model /path/to/sam3.pt"
            )
        self.model_path = model_path
        self.config_path = args.classes_config.resolve()
        self.classes, self.phrases = load_sam3_classes(self.config_path)
        self.names = {i: name for i, name in enumerate(self.classes)}
        self.dedup_iou = args.dedup_iou
        imgsz = args.imgsz or 1024
        overrides = dict(
            conf=args.conf,
            task="segment",
            mode="predict",
            model=str(model_path),
            imgsz=imgsz,
            device=args.device,
            half=args.half,
            save=False,
            verbose=False,
        )
        self.predictor = SAM3SemanticPredictor(overrides=overrides)

    def describe(self) -> str:
        pairs = ", ".join(f"{n} <- '{p}'" for n, p in zip(self.classes, self.phrases))
        return f"  config    : {self.config_path}\n  classes   : {pairs}"

    def infer(self, image_path: Path) -> dict:
        self.predictor.set_image(str(image_path))
        results = self.predictor(text=self.phrases)
        self.predictor.reset_image()
        return results_to_record(results[0], self.names, image_path, self.dedup_iou)


def main() -> None:
    args = parse_args()
    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    if not input_dir.exists():
        raise SystemExit(f"ERROR: input images folder not found: {input_dir}")

    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"ERROR: no images in {input_dir}. Drop images into input_images/ first.")

    backend = YoloBackend(args) if args.backend == "yolo" else Sam3Backend(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"prelabel_images (backend={args.backend})")
    print(f"  input dir : {input_dir}")
    print(f"  output dir: {output_dir}")
    print(f"  model     : {backend.model_path}")
    print(backend.describe())
    print(f"  conf      : {args.conf} | device: {args.device}")
    print(f"  dedup-iou : {args.dedup_iou} (0 = off; merges same-class duplicate masks)")
    print(f"  images    : {len(images)}")
    print()

    total_dets = 0
    for i, image_path in enumerate(images, 1):
        record = backend.infer(image_path)
        (output_dir / f"{image_path.stem}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        total_dets += len(record["detections"])
        print(f"  [{i}/{len(images)}] {image_path.name}: {len(record['detections'])} detections")

    print(f"\nDone. {total_dets} detections across {len(images)} images -> {output_dir}")
    print(f"Next: just export-{args.backend}")


if __name__ == "__main__":
    main()
