"""
Dataset QC report for a YOLO-seg dataset (label-level checks).

Runs two families of checks over DS/dataset_yolo (or any YOLO-seg root passed
with --root), reading class names from the dataset's data.yaml:

  1. Same-class duplicate labels  -> two polygons of the SAME class on the same
     object (high mask-IoU). This is what slips past box NMS during pre-labeling.
  2. Audit:
       - cross-class overlap (e.g. a cork and a pompom marked on one object),
       - degenerate polygons (<3 points or coords outside [0,1]),
       - near-empty masks (~0 area),
       - train/val leakage (same image stem in both splits).

Masks are rasterized from the normalized polygons onto a fixed grid, so IoU is
resolution-independent. Read-only: never modifies labels.

Usage:
  python utils/dataset_report.py
  python utils/dataset_report.py --root DS/dataset_yolo --dup-iou 0.7
  python utils/dataset_report.py --splits train val --top 25
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_ROOT = REPO_ROOT / "DS" / "dataset_yolo"


def load_names(root: Path) -> dict[int, str]:
    data_yaml = root / "data.yaml"
    if not data_yaml.is_file():
        return {}
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    names = data.get("names", {})
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def parse_label(path: str):
    """Return (instances, bad) where instance = (cls, line_no, points[(x,y)...])."""
    instances = []
    bad = []
    with open(path) as f:
        for ln, line in enumerate(f):
            parts = line.split()
            if not parts:
                continue
            cls = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
            npts = len(coords) // 2
            pts = [(coords[i], coords[i + 1]) for i in range(0, 2 * npts, 2)]
            instances.append((cls, ln, pts))
            if npts < 3:
                bad.append((ln, cls, f"poly<3pts({npts})"))
            elif any((x < -0.01 or x > 1.01 or y < -0.01 or y > 1.01) for x, y in pts):
                bad.append((ln, cls, "coords_out_of_[0,1]"))
    return instances, bad


def rasterize(pts, canvas: int) -> np.ndarray:
    img = Image.new("L", (canvas, canvas), 0)
    poly = [(x * canvas, y * canvas) for x, y in pts]
    if len(poly) >= 3:
        ImageDraw.Draw(img).polygon(poly, fill=1)
    return np.asarray(img, dtype=bool)


def iou_cont(ma: np.ndarray, mb: np.ndarray) -> tuple[float, float]:
    inter = int(np.logical_and(ma, mb).sum())
    if inter == 0:
        return 0.0, 0.0
    union = int(np.logical_or(ma, mb).sum())
    amin = int(min(ma.sum(), mb.sum()))
    return (inter / union if union else 0.0), (inter / amin if amin else 0.0)


def name_of(names: dict[int, str], cls: int):
    return names.get(cls, cls)


def analyze_split(root: Path, split: str, names: dict[int, str], canvas: int,
                  dup_iou: float, cross_iou: float, cross_cont: float, empty_area: int):
    ldir = root / "labels" / split
    files = sorted(glob.glob(str(ldir / "*.txt")))
    inst_count: dict[int, int] = {}
    dup_pairs = []     # (iou, cont, cls, fname, la, lb)
    cross_pairs = []   # (iou, cont, ca, cb, fname, la, lb)
    degenerate = []    # (fname, line, cls, why)
    empty = []         # (fname, line, cls, area)
    stems = set()

    for fp in files:
        fname = os.path.basename(fp)
        stems.add(os.path.splitext(fname)[0])
        instances, bad = parse_label(fp)
        for (ln, cls, why) in bad:
            degenerate.append((fname, ln, name_of(names, cls), why))

        masks = []
        for (cls, ln, pts) in instances:
            inst_count[cls] = inst_count.get(cls, 0) + 1
            m = rasterize(pts, canvas) if len(pts) >= 3 else None
            if m is not None and int(m.sum()) < empty_area:
                empty.append((fname, ln, name_of(names, cls), int(m.sum())))
            masks.append((cls, ln, m))

        for a in range(len(masks)):
            for b in range(a + 1, len(masks)):
                ca, la, ma = masks[a]
                cb, lb, mb = masks[b]
                if ma is None or mb is None:
                    continue
                iou, cont = iou_cont(ma, mb)
                if iou == 0 and cont == 0:
                    continue
                if ca == cb:
                    if iou >= dup_iou or cont >= 0.8:
                        dup_pairs.append((iou, cont, name_of(names, ca), fname, la, lb))
                else:
                    if iou >= cross_iou or cont >= cross_cont:
                        cross_pairs.append((iou, cont, name_of(names, ca), name_of(names, cb), fname, la, lb))

    return {
        "files": files, "inst_count": inst_count, "stems": stems,
        "dup_pairs": dup_pairs, "cross_pairs": cross_pairs,
        "degenerate": degenerate, "empty": empty,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="QC report for a YOLO-seg dataset (label-level checks).")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="YOLO-seg dataset root (has data.yaml, labels/, images/).")
    p.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to check.")
    p.add_argument("--dup-iou", type=float, default=0.7, help="Same-class mask-IoU above which two labels are a duplicate.")
    p.add_argument("--cross-iou", type=float, default=0.15, help="Cross-class mask-IoU to flag.")
    p.add_argument("--cross-cont", type=float, default=0.6, help="Cross-class containment to flag.")
    p.add_argument("--canvas", type=int, default=512, help="Rasterization grid size for IoU.")
    p.add_argument("--empty-area", type=int, default=9, help="Mask pixel area below this counts as near-empty.")
    p.add_argument("--top", type=int, default=25, help="Max offending pairs to list per category.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    if not (root / "labels").is_dir():
        raise SystemExit(f"ERROR: no labels/ under {root} (is this a YOLO-seg dataset root?)")
    names = load_names(root)

    print("=" * 72)
    print(f"Dataset QC report: {root}")
    print(f"Classes: {names or '(no data.yaml; showing raw class ids)'}")
    print(f"Thresholds: dup-IoU>={args.dup_iou} | cross-IoU>={args.cross_iou} | cross-cont>={args.cross_cont}")

    total_issues = 0
    for split in args.splits:
        if not (root / "labels" / split).is_dir():
            print(f"\n##### SPLIT {split}: (missing, skipped)")
            continue
        r = analyze_split(root, split, names, args.canvas, args.dup_iou,
                           args.cross_iou, args.cross_cont, args.empty_area)
        per_class = {name_of(names, c): r["inst_count"][c] for c in sorted(r["inst_count"])}
        print(f"\n##### SPLIT {split} ({len(r['files'])} label files)")
        print(f"  Instances per class: {per_class}")

        n_dup = len(r["dup_pairs"])
        n_cross = len(r["cross_pairs"])
        n_deg = len(r["degenerate"])
        n_empty = len(r["empty"])
        total_issues += n_dup + n_cross + n_deg + n_empty

        print(f"  [1] Same-class duplicate labels (IoU>={args.dup_iou}): {n_dup}")
        for (iou, cont, cls, fname, la, lb) in sorted(r["dup_pairs"], key=lambda p: -p[0])[: args.top]:
            print(f"       {cls:8s} IoU={iou:.2f} cont={cont:.2f} lines[{la},{lb}] {fname}")
        print(f"  [2] Cross-class overlap (IoU>={args.cross_iou} or cont>={args.cross_cont}): {n_cross}")
        for (iou, cont, ca, cb, fname, la, lb) in sorted(r["cross_pairs"], key=lambda p: -max(p[0], p[1]))[: args.top]:
            print(f"       {ca}<>{cb} IoU={iou:.2f} cont={cont:.2f} lines[{la},{lb}] {fname}")
        print(f"  [3] Degenerate polygons (<3 pts / coords outside [0,1]): {n_deg}")
        for d in r["degenerate"][: args.top]:
            print(f"       {d}")
        print(f"  [4] Near-empty masks (area<{args.empty_area}): {n_empty}")
        for e in r["empty"][: args.top]:
            print(f"       {e}")

    # cross-split leakage
    split_stems = {}
    for split in args.splits:
        ldir = root / "labels" / split
        if ldir.is_dir():
            split_stems[split] = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(str(ldir / "*.txt"))}
    leak = set()
    present = list(split_stems)
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            leak |= split_stems[present[i]] & split_stems[present[j]]
    total_issues += len(leak)
    print(f"\n##### Cross-split leakage (same stem in >1 split): {len(leak)}")
    for s in sorted(leak)[: args.top]:
        print(f"       {s}")

    print("\n" + "=" * 72)
    print(f"RESULT: {'CLEAN (0 issues)' if total_issues == 0 else f'{total_issues} issue(s) found'}")
    raise SystemExit(0 if total_issues == 0 else 1)


if __name__ == "__main__":
    main()
