"""Capa de visualización 2D (OpenCV): paneles RGB/depth con overlays de detecciones.

Solo presentación: no calcula poses. El núcleo (`pipeline.py`) no depende de este
módulo. `cv2` se importa de forma perezosa dentro de cada función, igual que en el
script original.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

# Per-class display colors live in config/class_colors.yaml (RGB), so anyone can
# tune them without touching code. Missing/extra classes just fall back to an
# auto-generated color, keeping the viewer agnostic to the class set.
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CLASS_COLORS_YAML = REPO_ROOT / "config" / "class_colors.yaml"


def _load_class_colors_bgr(path: Path = DEFAULT_CLASS_COLORS_YAML) -> dict[str, tuple[int, int, int]]:
    """Load per-class colors from YAML (RGB 0-255) as BGR tuples; {} on any error."""
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        out: dict[str, tuple[int, int, int]] = {}
        for name, rgb in (data.get("colors") or {}).items():
            r, g, b = (int(v) for v in rgb)
            out[str(name)] = (b, g, r)  # OpenCV uses BGR
        return out
    except Exception:
        return {}


CLASS_COLORS_BGR = _load_class_colors_bgr()


def color_for_class_bgr(class_name: str) -> tuple[int, int, int]:
    if class_name in CLASS_COLORS_BGR:
        return CLASS_COLORS_BGR[class_name]
    h = abs(hash(class_name)) % 360
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(h / 360.0, 0.65, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def colorize_depth(
    depth_m: np.ndarray,
    manual_lo: float = 0.0,
    manual_hi: float = 0.0,
) -> np.ndarray:
    import cv2

    valid = (depth_m > 0) & np.isfinite(depth_m)
    if not np.any(valid):
        return np.full(depth_m.shape + (3,), 25, dtype=np.uint8)
    if manual_lo > 0 and manual_hi > manual_lo:
        lo, hi = float(manual_lo), float(manual_hi)
    else:
        valid_values = depth_m[valid]
        lo = float(np.percentile(valid_values, 2))
        hi = float(np.percentile(valid_values, 98))
        if hi - lo < 0.05:
            mid = 0.5 * (lo + hi)
            lo, hi = max(0.01, mid - 0.05), mid + 0.05
    normalized = np.clip((depth_m - lo) / max(hi - lo, 1e-3), 0.0, 1.0)
    u8 = (normalized * 255.0).astype(np.uint8)
    try:
        colored = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    except Exception:
        colored = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
    colored[~valid] = (25, 25, 25)
    return colored


def fit_letterbox(
    img: np.ndarray,
    target_w: int,
    target_h: int,
    bg: tuple[int, int, int] = (16, 16, 16),
) -> tuple[np.ndarray, float, int, int]:
    import cv2

    target_w = max(1, int(target_w))
    target_h = max(1, int(target_h))
    h, w = img.shape[:2]
    canvas = np.full((target_h, target_w, 3), bg, dtype=np.uint8)
    if w == 0 or h == 0:
        return canvas, 1.0, 0, 0
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    ox = (target_w - new_w) // 2
    oy = (target_h - new_h) // 2
    canvas[oy : oy + new_h, ox : ox + new_w] = resized
    return canvas, scale, ox, oy


def render_rgb_panel(
    color_bgr: np.ndarray,
    detections: list[dict[str, Any]],
    fps: float,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    import cv2

    tinted = color_bgr.copy()
    for detection in detections:
        mask = detection.get("mask")
        if mask is None:
            continue
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.shape != color_bgr.shape[:2]:
            continue
        color = color_for_class_bgr(detection.get("class_name", ""))
        tint = np.array(color, dtype=np.float32)
        tinted[mask_bool] = (0.45 * tinted[mask_bool] + 0.55 * tint).astype(np.uint8)

    canvas, scale, ox, oy = fit_letterbox(tinted, target_w, target_h)
    line_thickness = max(2, int(round(scale * 2.0)))
    font_scale = max(0.55, min(1.6, scale * 0.65))
    text_thickness = max(1, int(round(font_scale * 2.0)))

    for detection in detections:
        bbox = detection.get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue
        x1, y1, x2, y2 = bbox
        sx1 = int(ox + x1 * scale)
        sy1 = int(oy + y1 * scale)
        sx2 = int(ox + x2 * scale)
        sy2 = int(oy + y2 * scale)
        color = color_for_class_bgr(detection.get("class_name", ""))
        cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), color, line_thickness)
        label = f"{detection.get('class_name','')} {float(detection.get('confidence',0.0)):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        pad = max(4, int(round(scale * 4)))
        bg_y1 = max(0, sy1 - th - 2 * pad)
        cv2.rectangle(canvas, (sx1, bg_y1), (sx1 + tw + 2 * pad, sy1), color, -1)
        cv2.putText(
            canvas,
            label,
            (sx1 + pad, sy1 - pad),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (10, 10, 10),
            text_thickness,
            cv2.LINE_AA,
        )

    fps_scale = max(0.7, min(1.6, scale * 0.7))
    fps_thickness = max(2, int(round(fps_scale * 2.0)))
    cv2.putText(
        canvas,
        f"{fps:.1f} FPS",
        (16, int(28 * fps_scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        fps_scale,
        (255, 255, 255),
        fps_thickness,
        cv2.LINE_AA,
    )
    return canvas


def render_depth_panel_canvas(
    depth_m: np.ndarray,
    target_w: int,
    target_h: int,
    manual_lo: float = 0.0,
    manual_hi: float = 0.0,
) -> np.ndarray:
    colored = colorize_depth(depth_m, manual_lo, manual_hi)
    canvas, _, _, _ = fit_letterbox(colored, target_w, target_h)
    return canvas


def compose_2d_view(
    color_bgr: np.ndarray,
    depth_m: np.ndarray,
    detections: list[dict[str, Any]],
    fps: float,
    layout: str,
    order: str,
    size_wh: tuple[int, int],
    depth_manual_lo: float = 0.0,
    depth_manual_hi: float = 0.0,
) -> np.ndarray:
    target_w, target_h = size_wh

    def rgb_at(tw: int, th: int) -> np.ndarray:
        return render_rgb_panel(color_bgr, detections, fps, tw, th)

    def depth_at(tw: int, th: int) -> np.ndarray:
        return render_depth_panel_canvas(depth_m, tw, th, depth_manual_lo, depth_manual_hi)

    if layout == "rgb-only":
        return rgb_at(target_w, target_h)
    if layout == "depth-only":
        return depth_at(target_w, target_h)
    if layout == "vertical":
        half_h = target_h // 2
        if order == "rgb-first":
            top, bot = rgb_at(target_w, half_h), depth_at(target_w, target_h - half_h)
        else:
            top, bot = depth_at(target_w, half_h), rgb_at(target_w, target_h - half_h)
        return np.vstack([top, bot])
    half_w = target_w // 2
    if order == "rgb-first":
        left, right = rgb_at(half_w, target_h), depth_at(target_w - half_w, target_h)
    else:
        left, right = depth_at(half_w, target_h), rgb_at(target_w - half_w, target_h)
    return np.hstack([left, right])
