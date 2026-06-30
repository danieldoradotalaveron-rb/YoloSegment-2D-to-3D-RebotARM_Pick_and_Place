"""Render synthetic RGB + instance_id views from labeled Gaussians (Phase 1B).

Backends:
  points  — z-sorted point splat (default, no extra deps)
  gsplat  — 3D Gaussian rasterization (requires: uv sync --extra 3dgs)

Usage:
  python DS/prelabel/render_synth_views.py
  python DS/prelabel/render_synth_views.py --capture capture_000004
  python DS/prelabel/render_synth_views.py --backend gsplat --capture capture_000004
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from labeled_gaussians import load_gaussians, load_gaussians_meta
from lift_rgbd import DEFAULT_RGBD_ROOT, list_labeled_capture_dirs, summarize_rgbd_captures
from synth_camera import default_virtual_views, project_points

DS_DIR = SCRIPT_DIR.parent
DEFAULT_SYNTH_ROOT = DS_DIR / "dataset_prelabel" / "synth_render_point"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render synthetic views from gaussians.npz. "
        "Default: all labeled RGBD captures that have gaussians.npz under synth-root."
    )
    parser.add_argument("--synth-root", type=Path, default=DEFAULT_SYNTH_ROOT)
    parser.add_argument(
        "--rgbd-root",
        type=Path,
        default=DEFAULT_RGBD_ROOT,
        help="Source RGBD root used to discover labeled captures (default: dataset_capture/rgbd).",
    )
    parser.add_argument(
        "--capture",
        default="",
        help="Single capture id (default: all labeled captures with gaussians.npz).",
    )
    parser.add_argument(
        "--backend",
        choices=("points", "gsplat"),
        default="points",
        help="points=splat lifted centers; gsplat=Gaussian raster (needs gsplat).",
    )
    parser.add_argument("--point-radius", type=int, default=2, help="Splat radius for points backend (px).")
    parser.add_argument("--views", type=int, default=0, help="Max views (0 = all defaults).")
    return parser.parse_args()


def render_points_view(
    xyz: np.ndarray,
    rgb01: np.ndarray,
    instance_id: np.ndarray,
    intrinsics: dict[str, Any],
    rotation: np.ndarray,
    width: int,
    height: int,
    point_radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    u, v, z = project_points(xyz, intrinsics, rotation)
    order = np.argsort(-z)
    rgb_out = np.zeros((height, width, 3), dtype=np.uint8)
    inst_out = np.zeros((height, width), dtype=np.uint16)
    r = max(1, point_radius)

    for idx in order:
        ui = int(round(u[idx]))
        vi = int(round(v[idx]))
        if ui < 0 or vi < 0 or ui >= width or vi >= height:
            continue
        if z[idx] <= 0:
            continue
        color = (rgb01[idx] * 255.0).astype(np.uint8)
        cv2.circle(rgb_out, (ui, vi), r, color.tolist(), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(inst_out, (ui, vi), r, int(instance_id[idx]), thickness=-1, lineType=cv2.LINE_8)

    return rgb_out, inst_out


def ensure_gsplat_ready() -> None:
    """Validate the gsplat backend once, with actionable errors, BEFORE any render.

    gsplat JIT-compiles CUDA kernels on first use, which needs the CUDA *toolkit*
    (nvcc + headers) — not just the driver. A tiny probe render is the ground truth:
    it surfaces a missing toolkit here so we never wipe existing outputs on failure.
    """
    try:
        import torch
        from gsplat import rasterization
    except ImportError as exc:
        raise SystemExit("gsplat backend requires: uv sync --extra 3dgs") from exc

    if not torch.cuda.is_available():
        raise SystemExit(
            "gsplat backend needs a CUDA GPU but torch.cuda.is_available() is False.\n"
            "  - Run on the machine with the GPU (not a CPU-only/sandboxed shell).\n"
            "  - Or use the CPU-friendly default: --backend points"
        )

    try:
        device = torch.device("cuda")
        means = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        quats = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        scales = torch.full((1, 3), 0.01, device=device)
        opacities = torch.ones((1,), device=device)
        colors = torch.ones((1, 3), device=device)
        viewmats = torch.eye(4, device=device).unsqueeze(0)
        ks = torch.tensor([[[10.0, 0.0, 4.0], [0.0, 10.0, 4.0], [0.0, 0.0, 1.0]]], device=device)
        rasterization(means, quats, scales, opacities, colors, viewmats, ks, 8, 8)
    except Exception as exc:  # noqa: BLE001 - any backend/build failure must be actionable
        raise SystemExit(
            "gsplat is installed but its CUDA backend failed to initialize:\n"
            f"  {type(exc).__name__}: {exc}\n\n"
            "Most likely the CUDA toolkit (nvcc) is missing. gsplat compiles its\n"
            "kernels on first use and needs nvcc matching the driver (CUDA 12.8),\n"
            "which is separate from torch.cuda being available.\n\n"
            "Fix options:\n"
            "  1) Install a prebuilt gsplat wheel for torch 2.10 + cu128, or\n"
            "  2) Install the CUDA toolkit (nvcc) matching the driver, then re-run, or\n"
            "  3) Use the default backend (no toolkit needed): --backend points"
        ) from exc


def render_gsplat_view(
    gaussians,
    intrinsics: dict[str, Any],
    rotation: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from gsplat import rasterization

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xyz = torch.from_numpy(gaussians.xyz).to(device)
    if rotation is not None:
        rot_t = torch.from_numpy(rotation.astype(np.float32)).to(device)
        xyz = (rot_t @ xyz.T).T

    scales = torch.exp(torch.from_numpy(gaussians.scales_log).to(device))
    quats = torch.from_numpy(gaussians.quats_wxyz).to(device)
    opacities = torch.from_numpy(gaussians.opacity).to(device)
    colors = torch.from_numpy(gaussians.rgb).to(device)

    viewmat = torch.eye(4, device=device, dtype=torch.float32)
    k = intrinsics
    kmat = torch.tensor(
        [[k["fx"], 0.0, k["ppx"]], [0.0, k["fy"], k["ppy"]], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    )
    viewmats = viewmat.unsqueeze(0)
    Ks = kmat.unsqueeze(0)

    render_colors, render_alphas, _ = rasterization(
        xyz,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        Ks,
        width,
        height,
    )
    rgb = (render_colors[0, ..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    # Instance buffer not supported in standard gsplat RGB pass; use points backend for instance_id.
    inst = np.zeros((height, width), dtype=np.uint16)
    return rgb, inst


def list_capture_dirs(synth_root: Path, rgbd_root: Path, capture_filter: str) -> list[Path]:
    synth_root = synth_root.resolve()
    rgbd_root = rgbd_root.resolve()
    capture_filter = capture_filter.strip()

    labeled_dirs = list_labeled_capture_dirs(rgbd_root, capture_filter)
    if capture_filter and not labeled_dirs:
        raise SystemExit(f"Labeled capture not found: {capture_filter} under {rgbd_root}")

    dirs: list[Path] = []
    missing_npz: list[str] = []
    for capture_dir in labeled_dirs:
        out_dir = synth_root / capture_dir.name
        if (out_dir / "gaussians.npz").is_file():
            dirs.append(out_dir)
        else:
            missing_npz.append(capture_dir.name)

    if missing_npz:
        print(f"SKIP render (no gaussians.npz): {', '.join(missing_npz)}")

    if not dirs:
        if capture_filter:
            raise SystemExit(f"No gaussians.npz for {capture_filter} under {synth_root}")
        labeled, unlabeled = summarize_rgbd_captures(rgbd_root, capture_filter)
        hint = f"{len(labeled)} labeled capture(s) under {rgbd_root}, "
        if not synth_root.is_dir():
            raise SystemExit(f"{hint}but {synth_root} does not exist (run just synth-render first)")
        raise SystemExit(f"{hint}but none have gaussians.npz in {synth_root} (run just synth-render first)")

    return dirs


def clean_stale_views(capture_dir: Path) -> None:
    """Remove previously rendered view artifacts so a rebuild is deterministic.

    Only touches regenerable render outputs; gaussians.npz/.meta.yaml are preserved.
    """
    prefix = capture_dir.name
    for pattern in (f"{prefix}_view_*.png", f"{prefix}_view_*.meta.yaml"):
        for stale in capture_dir.glob(pattern):
            stale.unlink()


def render_capture(capture_dir: Path, args: argparse.Namespace) -> None:
    npz_path = capture_dir / "gaussians.npz"
    gaussians = load_gaussians(npz_path)
    meta = load_gaussians_meta(npz_path)
    intrinsics = meta.get("intrinsics") or {}
    if not intrinsics:
        raise RuntimeError(f"{capture_dir.name}: missing intrinsics in gaussians.meta.yaml")

    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    views = default_virtual_views()
    if args.views > 0:
        views = views[: args.views]

    # RGB uses the chosen backend; the instance_id mask is always rasterized with
    # the points backend (gsplat's RGB pass has no per-gaussian id buffer).
    instance_backend = "points" if args.backend == "gsplat" else args.backend

    # Render everything to memory FIRST; only touch the disk once all views for
    # this capture succeed. This keeps a failed run from wiping good outputs.
    rendered: list[tuple[str, np.ndarray, np.ndarray, dict[str, Any]]] = []
    for view in views:
        view_id = view["view_id"]
        rotation = view["rotation"]
        if args.backend == "gsplat":
            rgb, inst = render_gsplat_view(gaussians, intrinsics, rotation, width, height)
            if not np.any(inst):
                _, inst = render_points_view(
                    gaussians.xyz,
                    gaussians.rgb,
                    gaussians.instance_id,
                    intrinsics,
                    rotation,
                    width,
                    height,
                    args.point_radius,
                )
        else:
            rgb, inst = render_points_view(
                gaussians.xyz,
                gaussians.rgb,
                gaussians.instance_id,
                intrinsics,
                rotation,
                width,
                height,
                args.point_radius,
            )
        stem = f"{capture_dir.name}_{view_id}"
        view_meta = {
            "capture_id": gaussians.capture_id,
            "view_id": view_id,
            "yaw_deg": view["yaw_deg"],
            "pitch_deg": view["pitch_deg"],
            "backend": args.backend,
            "instance_backend": instance_backend,
            "width": width,
            "height": height,
        }
        rendered.append((stem, rgb, inst, view_meta))

    clean_stale_views(capture_dir)
    for stem, rgb, inst, view_meta in rendered:
        rgb_path = capture_dir / f"{stem}.png"
        inst_path = capture_dir / f"{stem}.instance.png"
        meta_path = capture_dir / f"{stem}.meta.yaml"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(inst_path), inst)
        meta_path.write_text(
            yaml.safe_dump(view_meta, sort_keys=False),
            encoding="utf-8",
        )
        print(f"saved: {rgb_path.name}, {inst_path.name}")


def main() -> None:
    args = parse_args()
    if args.backend == "gsplat":
        ensure_gsplat_ready()
    synth_root = args.synth_root.resolve()
    rgbd_root = args.rgbd_root.resolve()
    capture_dirs = list_capture_dirs(synth_root, rgbd_root, args.capture.strip())
    print(f"Rendering {len(capture_dirs)} capture(s) [backend={args.backend}] from {synth_root}")

    ok: list[str] = []
    failed: list[tuple[str, str]] = []
    for capture_dir in capture_dirs:
        try:
            render_capture(capture_dir, args)
            ok.append(capture_dir.name)
        except (RuntimeError, OSError, ValueError) as exc:
            print(f"SKIP {capture_dir.name}: {exc}")
            failed.append((capture_dir.name, str(exc)))

    print(f"\n=== Render synth summary ===")
    print(f"OK: {len(ok)}/{len(capture_dirs)} capture(s)")
    for name in ok:
        print(f"  {name}")
    if failed:
        print(f"Skipped: {len(failed)}")
        for name, reason in failed:
            print(f"  {name}: {reason}")
    if not ok:
        raise SystemExit("No synthetic views rendered.")


if __name__ == "__main__":
    main()
