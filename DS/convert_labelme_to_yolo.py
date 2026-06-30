"""
Convert Labelme polygon JSON to Ultralytics YOLO segmentation layout.

Usage (from anywhere):
  python convert_labelme_to_yolo.py --data real      # baseline: only real
  python convert_labelme_to_yolo.py --data point      # real + point synth pipeline
  python convert_labelme_to_yolo.py --data 3dgs        # real + 3dgs synth pipeline
  python convert_labelme_to_yolo.py --data all         # real + point + 3dgs

Inputs:
  --data is REQUIRED (no default) so the dataset composition is always explicit.
  Real pool (always):   dataset_labeled/        (manual_labelme/, reviewed_*_labelme/)
  Point synth pool:     dataset_labeled_point/   (train-only; added by point|all)
  3dgs  synth pool:     dataset_labeled_3dgs/    (train-only; added by 3dgs|all)

  Each pool is scanned RECURSIVELY for .json + image pairs. The synthetic pools
  are PARALLEL pipelines that share image stems (capture_XXX_view_YYY), so on the
  YOLO side their output filenames are namespaced with a pipeline tag
  (point__<stem>, 3dgs__<stem>) to avoid collisions when --data all merges both.
  On-disk LabelMe files are never renamed (folders carry the suffix, not files).

Output (fully regenerated each run; NOT cumulative across runs):
  dataset_yolo/
    images/train|val/
    labels/train|val/
    data.yaml

Synthetic sources go ENTIRELY to train. The train/val split is computed on real
data only, so the validation set stays 100% real and the metric reflects
real-world performance.

Class ids come from the canonical DS/yolo_classes.yaml (ordered list = ids). If
that file is missing, ids fall back to sorted auto-discovery of the labels.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
# Real pool (shared by both pipelines): provides train AND val.
DEFAULT_REAL_INPUT = SCRIPT_DIR / "dataset_labeled"
# Synthetic pools (parallel, train-only). Selected explicitly via --data.
DEFAULT_POINT_INPUT = SCRIPT_DIR / "dataset_labeled_point"
DEFAULT_3DGS_INPUT = SCRIPT_DIR / "dataset_labeled_3dgs"
DEFAULT_OUTPUT = SCRIPT_DIR / "dataset_yolo"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
# Maps the --data choice to which synthetic pools (root, tag) to include.
SYNTH_POOLS = {
    "point": [("point", DEFAULT_POINT_INPUT)],
    "3dgs": [("3dgs", DEFAULT_3DGS_INPUT)],
    "all": [("point", DEFAULT_POINT_INPUT), ("3dgs", DEFAULT_3DGS_INPUT)],
    "real": [],
}
# Canonical class list (single source of truth for the YOLO pipeline). Pins the
# id<->name order so dataset/data.yaml/model stay consistent across conversions.
DEFAULT_CLASSES_YAML = SCRIPT_DIR / "yolo_classes.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Labelme JSON -> YOLO-seg dataset.")
    parser.add_argument(
        "--data",
        choices=["real", "point", "3dgs", "all"],
        required=True,
        help="Dataset composition (explicit, no default): real=only real; "
        "point/3dgs=real + that synth pipeline; all=real + point + 3dgs.",
    )
    parser.add_argument("--input-real", type=Path, default=DEFAULT_REAL_INPUT, help="Real pool (train+val).")
    parser.add_argument("--input-point", type=Path, default=DEFAULT_POINT_INPUT, help="Point synth pool (train-only).")
    parser.add_argument("--input-3dgs", type=Path, default=DEFAULT_3DGS_INPUT, help="3dgs synth pool (train-only).")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="YOLO dataset root.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Fraction for validation split.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train/val split.")
    parser.add_argument(
        "--classes-config",
        type=Path,
        default=DEFAULT_CLASSES_YAML,
        help="YAML with the canonical ordered class list (default: DS/yolo_classes.yaml).",
    )
    return parser.parse_args()


def synth_roots_for(args: argparse.Namespace) -> list[tuple[str, Path]]:
    """Resolve which (tag, root) synthetic pools to include from --data + overrides."""
    overrides = {"point": args.input_point, "3dgs": getattr(args, "input_3dgs")}
    pools: list[tuple[str, Path]] = []
    for tag, _default in SYNTH_POOLS[args.data]:
        pools.append((tag, overrides[tag]))
    return pools


def collect_class_names(json_files: list[Path]) -> list[str]:
    names: set[str] = set()
    for jf in json_files:
        data = json.loads(jf.read_text(encoding="utf-8"))
        for shape in data.get("shapes", []):
            label = shape.get("label", "").strip()
            if label:
                names.add(label)
    return sorted(names)


def load_canonical_classes(config_path: Path) -> list[str] | None:
    """Read the canonical ordered class list from YAML, or None if the file is absent.

    Accepts either a list of strings or a list of mappings with a `name` key.
    """
    if not config_path.is_file():
        return None
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    entries = data.get("classes")
    if not entries:
        raise SystemExit(f"ERROR: no 'classes' defined in {config_path}")
    names: list[str] = []
    for i, entry in enumerate(entries):
        if isinstance(entry, str):
            name = entry.strip()
        elif isinstance(entry, dict) and "name" in entry:
            name = str(entry["name"]).strip()
        else:
            raise SystemExit(f"ERROR: class #{i} in {config_path} must be a string or have a 'name'.")
        if name:
            names.append(name)
    return names


def resolve_class_names(config_path: Path, json_files: list[Path]) -> list[str]:
    """Pick class names from the canonical YAML (preferred) or fall back to discovery."""
    canonical = load_canonical_classes(config_path)
    discovered = set(collect_class_names(json_files))
    if canonical is None:
        print(f"WARNING: {config_path} not found; falling back to sorted auto-discovery of class ids.")
        return sorted(discovered)

    unknown = discovered - set(canonical)
    if unknown:
        raise SystemExit(
            f"ERROR: labels not declared in {config_path}: {sorted(unknown)}\n"
            f"       Add them to the 'classes:' list (append at the end to keep ids stable)."
        )
    return canonical


def collect_pairs(
    root: Path,
    output_dir: Path,
    tag: str | None,
) -> list[tuple[Path, Path, str]]:
    """Find (image, json, out_stem) triples under a pool root (recursive).

    out_stem is the basename used in the generated YOLO dataset. For synthetic
    pools it is namespaced with the pipeline tag (point__<stem>, 3dgs__<stem>) so
    the two pipelines never collide in dataset_yolo, even though their on-disk
    LabelMe files share the same stem. Real pool keeps its stem unchanged.
    """
    if not root.is_dir():
        return []
    items: list[tuple[Path, Path, str]] = []
    seen: set[str] = set()
    for jf in sorted(p for p in root.rglob("*.json") if output_dir not in p.parents):
        image = next((jf.with_suffix(s) for s in IMAGE_SUFFIXES if jf.with_suffix(s).exists()), None)
        if image is None:
            continue
        if jf.stem in seen:
            print(f"WARNING: duplicate image stem '{jf.stem}' under {root}; skipping {jf}")
            continue
        seen.add(jf.stem)
        out_stem = jf.stem if tag is None else f"{tag}__{jf.stem}"
        items.append((image, jf, out_stem))
    return items


def shape_to_yolo_line(shape: dict, class_to_id: dict[str, int], img_w: int, img_h: int) -> str | None:
    label = shape.get("label", "").strip()
    if not label or label not in class_to_id:
        return None

    shape_type = shape.get("shape_type", "")
    points = shape.get("points", [])
    if shape_type != "polygon" or len(points) < 3:
        return None

    coords: list[str] = []
    for x, y in points:
        nx = max(0.0, min(1.0, float(x) / img_w))
        ny = max(0.0, min(1.0, float(y) / img_h))
        coords.append(f"{nx:.6f}")
        coords.append(f"{ny:.6f}")

    class_id = class_to_id[label]
    return f"{class_id} " + " ".join(coords)


def labels_in_json(data: dict) -> set[str]:
    out: set[str] = set()
    for shape in data.get("shapes", []):
        lab = shape.get("label", "").strip()
        if lab:
            out.add(lab)
    return out


def stratified_split(
    pairs: list[tuple[Path, Path, str]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[Path, Path, str]], list[tuple[Path, Path, str]]]:
    """Shuffle split; try to keep at least one val image per class when possible."""
    rng = random.Random(seed)
    items = list(pairs)
    rng.shuffle(items)

    val_count = max(1, int(round(len(items) * val_ratio)))
    val_count = min(val_count, len(items) - 1) if len(items) > 1 else val_count

    val_set = items[:val_count]
    train_set = items[val_count:]

    all_classes: set[str] = set()
    for _img, jf, _stem in items:
        all_classes |= labels_in_json(json.loads(jf.read_text(encoding="utf-8")))

    val_classes: set[str] = set()
    for _img, jf, _stem in val_set:
        val_classes |= labels_in_json(json.loads(jf.read_text(encoding="utf-8")))

    missing = all_classes - val_classes
    for cls in sorted(missing):
        for i, (_img, jf, _stem) in enumerate(train_set):
            if cls in labels_in_json(json.loads(jf.read_text(encoding="utf-8"))):
                val_set.append(train_set.pop(i))
                val_classes.add(cls)
                break

    return train_set, val_set


def write_label_file(json_path: Path, label_path: Path, class_to_id: dict[str, int]) -> int:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    img_w = int(data.get("imageWidth") or 0)
    img_h = int(data.get("imageHeight") or 0)
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"{json_path.name}: invalid imageWidth/imageHeight")

    lines: list[str] = []
    for shape in data.get("shapes", []):
        line = shape_to_yolo_line(shape, class_to_id, img_w, img_h)
        if line:
            lines.append(line)

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def copy_split(
    split_name: str,
    pairs: list[tuple[Path, Path, str]],
    output: Path,
    class_to_id: dict[str, int],
) -> int:
    images_dir = output / "images" / split_name
    labels_dir = output / "labels" / split_name
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    instance_count = 0
    for image_path, json_path, out_stem in pairs:
        shutil.copy2(image_path, images_dir / f"{out_stem}{image_path.suffix}")
        instance_count += write_label_file(json_path, labels_dir / f"{out_stem}.txt", class_to_id)
    return instance_count


def write_data_yaml(output: Path, class_names: list[str]) -> Path:
    yaml_path = output / "data.yaml"
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    content = (
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names_block}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def main() -> None:
    args = parse_args()
    real_root = args.input_real.resolve()
    output_dir = args.output.resolve()

    # Real pool: always included, provides train AND val.
    real_pairs = collect_pairs(real_root, output_dir, tag=None)
    if not real_pairs:
        raise SystemExit(f"No .json/image pairs under real pool {real_root}")

    # Synthetic pools: train-only, selected explicitly by --data.
    synth_pairs: list[tuple[Path, Path, str]] = []
    synth_summary: list[str] = []
    for tag, root in synth_roots_for(args):
        pool = collect_pairs(root.resolve(), output_dir, tag=tag)
        if not pool:
            print(f"WARNING: synth pool '{tag}' empty or missing: {root.resolve()}")
        synth_pairs += pool
        synth_summary.append(f"{tag}={len(pool)}")

    all_pairs = real_pairs + synth_pairs
    class_names = resolve_class_names(args.classes_config.resolve(), [jf for _img, jf, _stem in all_pairs])
    class_to_id = {name: i for i, name in enumerate(class_names)}

    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Split only the real data; the validation set stays 100% real. Synthetic
    # (train-only) sources are appended to train so they never leak into val.
    train_pairs, val_pairs = stratified_split(real_pairs, args.val_ratio, args.seed)
    train_pairs = train_pairs + synth_pairs

    train_instances = copy_split("train", train_pairs, output_dir, class_to_id)
    val_instances = copy_split("val", val_pairs, output_dir, class_to_id)
    yaml_path = write_data_yaml(output_dir, class_names)

    sources = "real" if not synth_pairs else "real + " + ", ".join(synth_summary)
    print(f"Data mode (--data {args.data}): {sources}")
    print(f"Real pool:  {real_root}")
    for tag, root in synth_roots_for(args):
        print(f"Synth pool [{tag}]: {root.resolve()}")
    print(f"Output: {output_dir}")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    print(f"Train: {len(train_pairs)} images, {train_instances} instances")
    print(f"  (real: {len(train_pairs) - len(synth_pairs)}, synthetic train-only: {len(synth_pairs)})")
    print(f"Val:   {len(val_pairs)} images, {val_instances} instances (100% real)")
    print(f"data.yaml: {yaml_path}")
    print()
    repo_root = SCRIPT_DIR.parent
    print("Train (imgsz=448, same as TabletopSeg):")
    print(f"  cd {repo_root}\n  just train")


if __name__ == "__main__":
    main()
