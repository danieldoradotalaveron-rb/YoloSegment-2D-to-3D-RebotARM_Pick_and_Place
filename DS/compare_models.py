"""Validate every trained best.pt against the SAME (current) val split and tabulate.

Why: each runs/segment/<run>/results.csv was computed on the dataset that existed
when that run trained (the dataset has grown over time, and convert re-splits
train/val on every run). Those numbers are therefore NOT comparable across runs.
This script re-runs `model.val()` for all weights on the current data.yaml, so the
metrics are apples-to-apples, and prints a ranked table (+ optional CSV).

Usage:
  python DS/compare_models.py
  python DS/compare_models.py --data DS/dataset_yolo/data.yaml --imgsz 448
  python DS/compare_models.py --runs-root runs/segment --csv runs/segment/model_comparison.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_RUNS_ROOT = REPO_ROOT / "runs" / "segment"
DEFAULT_DATA = SCRIPT_DIR / "dataset_yolo" / "data.yaml"

# (results_dict key, short column header). Box (B) and Mask (M) metrics.
METRIC_COLUMNS = [
    ("metrics/mAP50-95(M)", "mask_mAP50-95"),
    ("metrics/mAP50(M)", "mask_mAP50"),
    ("metrics/mAP50-95(B)", "box_mAP50-95"),
    ("metrics/mAP50(B)", "box_mAP50"),
    ("metrics/precision(M)", "mask_P"),
    ("metrics/recall(M)", "mask_R"),
]
RANK_KEY = "metrics/mAP50-95(M)"  # mask mAP50-95 is the strict, stable default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare trained YOLO-seg models on the current val split.")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--imgsz", type=int, default=448)
    parser.add_argument("--device", default="0", help="cuda id or 'cpu' (default: 0).")
    parser.add_argument("--csv", type=Path, default=DEFAULT_RUNS_ROOT / "model_comparison.csv")
    return parser.parse_args()


def find_weights(runs_root: Path) -> list[Path]:
    runs_root = runs_root.resolve()
    if not runs_root.is_dir():
        raise SystemExit(f"Not found: {runs_root} (train a model first)")
    weights = sorted(runs_root.glob("*/weights/best.pt"))
    if not weights:
        raise SystemExit(f"No best.pt under {runs_root}/*/weights/")
    return weights


def run_name(weight: Path) -> str:
    # runs/segment/<run>/weights/best.pt -> <run>
    return weight.parent.parent.name


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(f"Not found: {args.data} (run 'just convert --data real|point|3dgs|all' first)")

    from ultralytics import YOLO  # imported late so --help works without the dep

    weights = find_weights(args.runs_root)
    print(f"Comparing {len(weights)} model(s) on {args.data} (imgsz={args.imgsz}, device={args.device})\n")

    rows: list[dict[str, object]] = []
    for weight in weights:
        name = run_name(weight)
        print(f"-> validating {name} ...")
        model = YOLO(str(weight))
        metrics = model.val(
            data=str(args.data),
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
            plots=False,
            project=str(args.runs_root / "_compare"),
            name=name,
            exist_ok=True,
        )
        rd = metrics.results_dict
        row: dict[str, object] = {"model": name}
        for key, header in METRIC_COLUMNS:
            row[header] = round(float(rd.get(key, float("nan"))), 4)
        row["_rank"] = float(rd.get(RANK_KEY, float("nan")))
        rows.append(row)

    rows.sort(key=lambda r: (r["_rank"] != r["_rank"], -float(r["_rank"])))  # NaN last, desc

    headers = ["model"] + [h for _, h in METRIC_COLUMNS]
    widths = {h: max(len(h), *(len(f"{r[h]}") for r in rows)) for h in headers}
    sep = "  "
    line = sep.join(h.ljust(widths[h]) for h in headers)
    print(f"\n{line}")
    print(sep.join("-" * widths[h] for h in headers))
    for i, r in enumerate(rows):
        cells = [str(r[h]).ljust(widths[h]) for h in headers]
        mark = "  <- best" if i == 0 else ""
        print(sep.join(cells) + mark)

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", encoding="utf-8") as fh:
            fh.write(",".join(headers) + "\n")
            for r in rows:
                fh.write(",".join(f"{r[h]}" for h in headers) + "\n")
        print(f"\nSaved: {args.csv}")
    print(f"\nRanked by {RANK_KEY} (mask mAP50-95). Per-model val output: {args.runs_root / '_compare'}/")


if __name__ == "__main__":
    main()
