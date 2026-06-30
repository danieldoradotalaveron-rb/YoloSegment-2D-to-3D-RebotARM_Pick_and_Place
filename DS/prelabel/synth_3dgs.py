"""Phase 1B: init labeled Gaussians + render synthetic views (orchestrator).

Single entry point behind `just synth-render` (runs init + render in 2 steps).
The backend selects the rasterizer AND, via sense.sh, the output folder suffix:
  --backend points -> synth_render_point/   --backend gsplat -> synth_render_3dgs/

Usage:
  python DS/prelabel/synth_3dgs.py --inpaint-depth --synth-root .../synth_render_point
  python DS/prelabel/synth_3dgs.py --capture capture_000004 --inpaint-depth
  python DS/prelabel/synth_3dgs.py --inpaint-depth --backend gsplat
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lift_rgbd import DEFAULT_RGBD_ROOT, summarize_rgbd_captures

DS_DIR = SCRIPT_DIR.parent
DEFAULT_SYNTH_ROOT = DS_DIR / "dataset_prelabel" / "synth_render_point"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Init gaussians + render synthetic views. "
        "Default: all labeled captures under dataset_capture/rgbd."
    )
    parser.add_argument(
        "--capture",
        default="",
        help="Single capture id (default: all labeled captures).",
    )
    parser.add_argument("--rgbd-root", type=Path, default=DEFAULT_RGBD_ROOT)
    parser.add_argument("--synth-root", type=Path, default=DEFAULT_SYNTH_ROOT)
    parser.add_argument("--inpaint-depth", action="store_true")
    parser.add_argument("--backend", choices=("points", "gsplat"), default="points")
    parser.add_argument("--views", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rgbd_root = args.rgbd_root.resolve()
    labeled, unlabeled = summarize_rgbd_captures(rgbd_root, args.capture.strip())
    if not labeled:
        raise SystemExit(f"No labeled captures (rgb.json) under {rgbd_root}")

    scope = args.capture.strip() or f"all {len(labeled)} labeled capture(s)"
    print(f"synth-render [{args.backend}]: {scope} under {rgbd_root}")
    print(f"  output -> {args.synth_root.resolve()}")
    if unlabeled and not args.capture.strip():
        names = ", ".join(d.name for d in unlabeled)
        print(f"  (skipping {len(unlabeled)} without labels: {names})")

    py = sys.executable
    init_cmd = [
        py,
        str(SCRIPT_DIR / "init_labeled_gaussians.py"),
        "--rgbd-root",
        str(rgbd_root),
        "--output-root",
        str(args.synth_root.resolve()),
    ]
    render_cmd = [
        py,
        str(SCRIPT_DIR / "render_synth_views.py"),
        "--rgbd-root",
        str(rgbd_root),
        "--synth-root",
        str(args.synth_root.resolve()),
        "--backend",
        args.backend,
    ]
    if args.capture.strip():
        init_cmd += ["--capture", args.capture.strip()]
        render_cmd += ["--capture", args.capture.strip()]
    if args.inpaint_depth:
        init_cmd.append("--inpaint-depth")
    if args.views > 0:
        render_cmd += ["--views", str(args.views)]

    print("\n[paso 1/2] init-gaussians: lift RGBD -> gaussians.npz")
    subprocess.run(init_cmd, check=True)
    print(f"\n[paso 2/2] render: gaussians.npz -> vistas sintéticas [backend={args.backend}]")
    subprocess.run(render_cmd, check=True)
    print("\nsynth-render OK (init + render completados).")


if __name__ == "__main__":
    main()
