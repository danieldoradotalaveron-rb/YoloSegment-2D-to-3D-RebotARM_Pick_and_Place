"""
Analyze Labelme and YOLO-seg datasets; compare conversion consistency.

Usage:
  python analyze_dataset.py                  # all reports
  python analyze_dataset.py --mode labelme
  python analyze_dataset.py --mode yolo
  python analyze_dataset.py --mode compare
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# Human-owned pool of finished LabelMe annotations (scanned recursively).
DEFAULT_LABELME = SCRIPT_DIR / "dataset_labeled"
DEFAULT_YOLO = SCRIPT_DIR / "dataset_yolo"
YOLO_SPLITS = ("train", "val")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageRecord:
    stem: str
    instances: int = 0
    class_counts: Counter = field(default_factory=Counter)
    size: tuple[int, int] | None = None
    split: str | None = None
    shape_types: Counter = field(default_factory=Counter)
    skipped_shapes: int = 0


@dataclass
class DatasetSummary:
    root: Path
    records: dict[str, ImageRecord] = field(default_factory=dict)
    class_names: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    @property
    def image_count(self) -> int:
        return len(self.records)

    @property
    def instance_count(self) -> int:
        return sum(r.instances for r in self.records.values())

    def class_instance_counts(self) -> Counter:
        total: Counter = Counter()
        for rec in self.records.values():
            total.update(rec.class_counts)
        return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Labelme / YOLO datasets.")
    parser.add_argument(
        "--mode",
        choices=("labelme", "yolo", "compare", "all"),
        default="all",
        help="Which report to run (default: all).",
    )
    parser.add_argument("--labelme-dir", type=Path, default=DEFAULT_LABELME)
    parser.add_argument("--yolo-dir", type=Path, default=DEFAULT_YOLO)
    return parser.parse_args()


def _image_path_in_dir(directory: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _read_image_size(image_path: Path) -> tuple[int, int] | None:
    try:
        import cv2

        img = cv2.imread(str(image_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        return (w, h)
    except Exception:
        return None


# --- Labelme ---


def parse_labelme_json(json_path: Path) -> ImageRecord:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    w = int(data.get("imageWidth") or 0)
    h = int(data.get("imageHeight") or 0)
    size = (w, h) if w > 0 and h > 0 else None

    rec = ImageRecord(stem=json_path.stem, size=size)
    for shape in data.get("shapes", []):
        shape_type = shape.get("shape_type", "?")
        rec.shape_types[shape_type] += 1
        label = shape.get("label", "").strip()
        points = shape.get("points", [])

        if shape_type == "polygon" and label and len(points) >= 3:
            rec.instances += 1
            rec.class_counts[label] += 1
        elif label or points:
            rec.skipped_shapes += 1

    return rec


def scan_labelme(labelme_dir: Path) -> DatasetSummary:
    summary = DatasetSummary(root=labelme_dir.resolve())
    if not labelme_dir.is_dir():
        summary.issues.append(f"directory not found: {labelme_dir}")
        return summary

    json_files = sorted(labelme_dir.rglob("*.json"))
    class_names: set[str] = set()

    for jf in json_files:
        rec = parse_labelme_json(jf)
        if rec.stem in summary.records:
            summary.issues.append(f"{rec.stem}: duplicate stem across subfolders (using first)")
            continue
        summary.records[rec.stem] = rec
        class_names.update(rec.class_counts.keys())

        jpg = _image_path_in_dir(jf.parent, rec.stem)
        if jpg is None:
            summary.issues.append(f"{rec.stem}: json without matching image")
        elif rec.size is None:
            size = _read_image_size(jpg)
            if size:
                rec.size = size

    jpgs = [p for p in labelme_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES]
    json_stems = {p.stem for p in json_files}
    for img in jpgs:
        if img.stem not in json_stems:
            summary.issues.append(f"{img.name}: image without json")

    summary.class_names = sorted(class_names)
    return summary


def print_labelme_report(summary: DatasetSummary) -> None:
    print("=" * 60)
    print("LABELME", summary.root)
    print("=" * 60)

    if summary.issues:
        print("\n=== ISSUES ===")
        for issue in summary.issues:
            print(f"  ! {issue}")

    json_count = len(list(summary.root.rglob("*.json")))
    img_count = len([p for p in summary.root.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES])
    print("\n=== COUNTS ===")
    print(f"images: {img_count}")
    print(f"json:   {json_count}")
    print(f"paired stems (in report): {summary.image_count}")

    sizes = Counter(r.size for r in summary.records.values() if r.size)
    print("\n=== IMAGE SIZE ===")
    for size, count in sizes.most_common():
        print(f"  {size}: {count}")

    print("\n=== CLASSES (instances) ===")
    for name, count in summary.class_instance_counts().most_common():
        print(f"  {name!r}: {count}")

    shape_types: Counter = Counter()
    for rec in summary.records.values():
        shape_types.update(rec.shape_types)
    print("\n=== SHAPE TYPES ===")
    for t, count in shape_types.most_common():
        print(f"  {t}: {count}")

    objs = [r.instances for r in summary.records.values()]
    empty = sum(1 for n in objs if n == 0)
    print("\n=== ANNOTATIONS PER IMAGE ===")
    print(f"  empty (0 valid polygons): {empty}")
    if objs:
        print(f"  min/max/mean: {min(objs)}/{max(objs)}/{sum(objs)/len(objs):.2f}")

    combo: Counter = Counter()
    for rec in summary.records.values():
        combo[tuple(sorted(rec.class_counts.keys()))] += 1
    print("\n=== CLASS COMBOS (images with >=1 instance) ===")
    for combo_key, count in combo.most_common():
        label = combo_key if combo_key else "(empty)"
        print(f"  {label}: {count}")

    print("\n=== OBJECTS PER IMAGE HISTOGRAM ===")
    for k in sorted(Counter(objs)):
        print(f"  {k} objects: {Counter(objs)[k]} images")


# --- YOLO ---


def load_yolo_class_names(yolo_dir: Path) -> list[str]:
    yaml_path = yolo_dir / "data.yaml"
    if not yaml_path.exists():
        return []

    id_to_name: dict[int, str] = {}
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key, value = key.strip(), value.strip().strip("'\"")
        if key.isdigit() and value:
            id_to_name[int(key)] = value
    return [id_to_name[i] for i in sorted(id_to_name)]


def parse_yolo_label(label_path: Path, class_names: list[str]) -> ImageRecord:
    rec = ImageRecord(stem=label_path.stem)
    if not label_path.exists() or label_path.stat().st_size == 0:
        return rec

    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 7:
            rec.skipped_shapes += 1
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            rec.skipped_shapes += 1
            continue
        coords = parts[1:]
        if len(coords) % 2 != 0:
            rec.skipped_shapes += 1
            continue

        name = class_names[class_id] if class_id < len(class_names) else f"id_{class_id}"
        rec.instances += 1
        rec.class_counts[name] += 1

    return rec


def scan_yolo(yolo_dir: Path) -> DatasetSummary:
    summary = DatasetSummary(root=yolo_dir.resolve())
    if not yolo_dir.is_dir():
        summary.issues.append(f"directory not found: {yolo_dir}")
        return summary

    class_names = load_yolo_class_names(yolo_dir)
    summary.class_names = class_names

    yaml_path = yolo_dir / "data.yaml"
    if not yaml_path.exists():
        summary.issues.append("data.yaml not found")

    for split in YOLO_SPLITS:
        images_dir = yolo_dir / "images" / split
        labels_dir = yolo_dir / "labels" / split
        if not images_dir.is_dir():
            summary.issues.append(f"missing images/{split}")
            continue
        if not labels_dir.is_dir():
            summary.issues.append(f"missing labels/{split}")
            continue

        for img_path in sorted(images_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            stem = img_path.stem
            label_path = labels_dir / f"{stem}.txt"
            if stem in summary.records:
                summary.issues.append(f"duplicate stem across splits: {stem}")
                continue

            rec = parse_yolo_label(label_path, class_names)
            rec.stem = stem
            rec.split = split
            rec.size = _read_image_size(img_path)
            summary.records[stem] = rec

            if not label_path.exists():
                summary.issues.append(f"{stem} ({split}): image without label file")
            elif rec.instances == 0 and label_path.stat().st_size > 0:
                summary.issues.append(f"{stem} ({split}): label file has no valid lines")

        label_files = sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []
        image_stems = {p.stem for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
        for lf in label_files:
            if lf.stem not in image_stems:
                summary.issues.append(f"{lf.name} ({split}): label without image")

    discovered: set[str] = set()
    for rec in summary.records.values():
        discovered.update(rec.class_counts.keys())
    if class_names:
        missing_in_yaml = discovered - set(class_names)
        if missing_in_yaml:
            summary.issues.append(f"labels use unknown classes: {sorted(missing_in_yaml)}")

    return summary


def print_yolo_report(summary: DatasetSummary) -> None:
    print("=" * 60)
    print("YOLO-SEG", summary.root)
    print("=" * 60)

    if summary.issues:
        print("\n=== ISSUES ===")
        for issue in summary.issues:
            print(f"  ! {issue}")

    print("\n=== data.yaml CLASSES ===")
    if summary.class_names:
        for i, name in enumerate(summary.class_names):
            print(f"  {i}: {name}")
    else:
        print("  (not found)")

    split_counts = Counter(r.split for r in summary.records.values())
    print("\n=== SPLITS ===")
    for split in YOLO_SPLITS:
        n = split_counts.get(split, 0)
        inst = sum(r.instances for r in summary.records.values() if r.split == split)
        print(f"  {split}: {n} images, {inst} instances")

    print("\n=== TOTAL ===")
    print(f"images: {summary.image_count}")
    print(f"instances: {summary.instance_count}")

    sizes = Counter(r.size for r in summary.records.values() if r.size)
    print("\n=== IMAGE SIZE ===")
    for size, count in sizes.most_common():
        print(f"  {size}: {count}")

    print("\n=== CLASSES (instances) ===")
    for name, count in summary.class_instance_counts().most_common():
        print(f"  {name!r}: {count}")

    objs = [r.instances for r in summary.records.values()]
    empty = sum(1 for n in objs if n == 0)
    print("\n=== ANNOTATIONS PER IMAGE ===")
    print(f"  empty label files: {empty}")
    if objs:
        print(f"  min/max/mean: {min(objs)}/{max(objs)}/{sum(objs)/len(objs):.2f}")

    combo: Counter = Counter()
    for rec in summary.records.values():
        combo[tuple(sorted(rec.class_counts.keys()))] += 1
    print("\n=== CLASS COMBOS ===")
    for combo_key, count in combo.most_common():
        label = combo_key if combo_key else "(empty)"
        print(f"  {label}: {count}")


# --- Compare ---


def compare_datasets(labelme: DatasetSummary, yolo: DatasetSummary) -> None:
    print("=" * 60)
    print("COMPARE  labelme  <->  yolo")
    print("=" * 60)

    lm_stems = set(labelme.records)
    yolo_stems = set(yolo.records)

    only_labelme = sorted(lm_stems - yolo_stems)
    only_yolo = sorted(yolo_stems - lm_stems)
    common = sorted(lm_stems & yolo_stems)

    print("\n=== COVERAGE ===")
    print(f"labelme stems: {len(lm_stems)}")
    print(f"yolo stems:    {len(yolo_stems)}")
    print(f"in both:       {len(common)}")
    print(f"only labelme (not converted yet): {len(only_labelme)}")
    if only_labelme:
        preview = ", ".join(only_labelme[:8])
        suffix = " ..." if len(only_labelme) > 8 else ""
        print(f"  e.g. {preview}{suffix}")
    print(f"only yolo (unexpected): {len(only_yolo)}")
    if only_yolo:
        print(f"  {only_yolo[:10]}")

    lm_classes = set(labelme.class_names) | set(labelme.class_instance_counts())
    yolo_classes = set(yolo.class_names) | set(yolo.class_instance_counts())
    print("\n=== CLASSES ===")
    print(f"labelme: {sorted(lm_classes)}")
    print(f"yolo:    {sorted(yolo_classes)}")
    only_lm_cls = sorted(lm_classes - yolo_classes)
    only_yolo_cls = sorted(yolo_classes - lm_classes)
    if only_lm_cls:
        print(f"  only in labelme: {only_lm_cls}")
    if only_yolo_cls:
        print(f"  only in yolo: {only_yolo_cls}")

    lm_inst = labelme.class_instance_counts()
    yolo_inst = yolo.class_instance_counts()
    all_cls = sorted(lm_classes | yolo_classes)
    print("\n=== INSTANCE COUNTS BY CLASS ===")
    print(f"  {'class':<12} {'labelme':>8} {'yolo':>8} {'delta':>8}")
    mismatch_cls = False
    for cls in all_cls:
        a, b = lm_inst.get(cls, 0), yolo_inst.get(cls, 0)
        delta = b - a
        flag = "  !" if delta != 0 else ""
        if delta != 0:
            mismatch_cls = True
        print(f"  {cls:<12} {a:8} {b:8} {delta:+8}{flag}")

    print("\n=== GLOBAL INSTANCES ===")
    print(f"  labelme: {labelme.instance_count}")
    print(f"  yolo:    {yolo.instance_count}")
    print(f"  delta:   {yolo.instance_count - labelme.instance_count:+}")

    inst_mismatch: list[str] = []
    class_mismatch: list[str] = []
    for stem in common:
        lm_rec = labelme.records[stem]
        yo_rec = yolo.records[stem]
        if lm_rec.instances != yo_rec.instances:
            inst_mismatch.append(
                f"{stem}: labelme={lm_rec.instances} yolo={yo_rec.instances} ({yo_rec.split})"
            )
        if lm_rec.class_counts != yo_rec.class_counts:
            class_mismatch.append(
                f"{stem}: labelme={dict(lm_rec.class_counts)} yolo={dict(yo_rec.class_counts)}"
            )

    print("\n=== PER-IMAGE INSTANCE COUNT ===")
    print(f"  matching:  {len(common) - len(inst_mismatch)}")
    print(f"  mismatches: {len(inst_mismatch)}")
    for line in inst_mismatch[:15]:
        print(f"    ! {line}")
    if len(inst_mismatch) > 15:
        print(f"    ... and {len(inst_mismatch) - 15} more")

    print("\n=== PER-IMAGE CLASS COUNTS ===")
    print(f"  matching:  {len(common) - len(class_mismatch)}")
    print(f"  mismatches: {len(class_mismatch)}")
    for line in class_mismatch[:10]:
        print(f"    ! {line}")
    if len(class_mismatch) > 10:
        print(f"    ... and {len(class_mismatch) - 10} more")

    unconverted_with_labels = [
        s
        for s in only_labelme
        if labelme.records[s].instances > 0
    ]
    print("\n=== ACTION HINTS ===")
    if only_labelme:
        print(f"  Re-run convert_labelme_to_yolo.py to include {len(only_labelme)} new image(s).")
    if mismatch_cls or inst_mismatch:
        print("  Instance totals differ: re-convert or check skipped polygon shapes in labelme.")
    if not only_labelme and not inst_mismatch and not mismatch_cls:
        print("  Datasets are consistent for converted images.")

    if yolo.image_count > 0 and labelme.image_count > 0:
        ratio = yolo.image_count / labelme.image_count
        print(f"\n  yolo uses {yolo.image_count}/{labelme.image_count} labelme images ({ratio:.0%})")


def main() -> None:
    args = parse_args()
    labelme_dir = args.labelme_dir.resolve()
    yolo_dir = args.yolo_dir.resolve()

    labelme_summary: DatasetSummary | None = None
    yolo_summary: DatasetSummary | None = None

    if args.mode in ("labelme", "all", "compare"):
        labelme_summary = scan_labelme(labelme_dir)
    if args.mode in ("yolo", "all", "compare"):
        yolo_summary = scan_yolo(yolo_dir)

    if args.mode in ("labelme", "all") and labelme_summary:
        print_labelme_report(labelme_summary)
        print()

    if args.mode in ("yolo", "all") and yolo_summary:
        print_yolo_report(yolo_summary)
        print()

    if args.mode in ("compare", "all") and labelme_summary and yolo_summary:
        compare_datasets(labelme_summary, yolo_summary)


if __name__ == "__main__":
    main()
