"""Visualize a depth.npy map saved by capture.py (RGBD mode).

Examples:
  python DS/visualize_depth.py --depth-file DS/dataset_capture/rgbd/capture_000001/depth.npy
  python DS/visualize_depth.py --depth-file .../depth.npy --save preview.png
  python DS/visualize_depth.py --depth-file .../depth.npy --min 0.1 --max 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def colorize_depth(
    depth_m: np.ndarray,
    depth_min: float | None,
    depth_max: float | None,
) -> np.ndarray:
    """Map metric depth to a BGR JET image. Invalid (<=0) pixels render black."""
    valid = depth_m > 0
    if not np.any(valid):
        return np.zeros((*depth_m.shape, 3), dtype=np.uint8)

    lo = depth_min if depth_min is not None else float(depth_m[valid].min())
    hi = depth_max if depth_max is not None else float(depth_m[valid].max())
    if hi <= lo:
        hi = lo + 1e-6

    normalized = np.clip((depth_m - lo) / (hi - lo), 0.0, 1.0)
    gray = (normalized * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    colored[~valid] = (0, 0, 0)
    return colored


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a depth.npy map.")
    parser.add_argument("--depth-file", required=True, help="Path to depth.npy (float32, meters).")
    parser.add_argument("--min", type=float, default=None, help="Min depth (m) for color scale. Default: data min.")
    parser.add_argument("--max", type=float, default=None, help="Max depth (m) for color scale. Default: data max.")
    parser.add_argument("--save", default="", help="Write the colorized PNG to this path (no GUI needed).")
    parser.add_argument(
        "--rgb",
        action="store_true",
        help="Show rgb.png from the same folder side by side (if present).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    depth_path = Path(args.depth_file)
    if not depth_path.is_file():
        raise SystemExit(f"Not found: {depth_path}")

    depth_m = np.load(depth_path).astype(np.float32)
    if depth_m.ndim != 2:
        raise SystemExit(f"Expected a 2D depth map, got shape {depth_m.shape}")

    valid = depth_m > 0
    n_valid = int(valid.sum())
    pct = 100.0 * n_valid / depth_m.size if depth_m.size else 0.0
    print(f"file:   {depth_path}")
    print(f"shape:  {depth_m.shape}  dtype: {depth_m.dtype}")
    print(f"valid:  {n_valid}/{depth_m.size} ({pct:.1f}%)")
    if n_valid:
        v = depth_m[valid]
        print(f"depth m: min={v.min():.3f}  median={np.median(v):.3f}  max={v.max():.3f}")

    colored = colorize_depth(depth_m, args.min, args.max)

    panel = colored
    if args.rgb:
        rgb_path = depth_path.parent / "rgb.png"
        rgb = cv2.imread(str(rgb_path))
        if rgb is not None:
            if rgb.shape[:2] != colored.shape[:2]:
                rgb = cv2.resize(rgb, (colored.shape[1], colored.shape[0]))
            panel = np.hstack([rgb, colored])
        else:
            print(f"(rgb requested but not found: {rgb_path})")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), panel)
        print(f"saved:  {out}")
        return

    window = f"depth: {depth_path.name}"
    cv2.imshow(window, panel)
    print("Press any key (or q) on the window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
