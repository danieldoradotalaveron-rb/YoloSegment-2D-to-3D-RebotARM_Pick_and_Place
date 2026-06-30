"""Generate a ChArUco board PNG sized to fill an e-ink / LCD screen 1:1.

Defaults target a BOOX Go 10.3 (1872x1404 px, ~227 ppi). The output canvas matches
the panel resolution exactly, so opening it full-screen ("fit to screen") shows it 1:1
with no scaling. ALWAYS verify the printed/displayed square edge with a ruler and use
that measured value in the calibration: the real square size sets the translation scale.

Usage:
    uv run python handeye/calibration_d405/generate_charuco.py
    uv run python handeye/calibration_d405/generate_charuco.py --panel-w 1872 --panel-h 1404 --ppi 227 \
        --squares-x 8 --squares-y 6 --square-mm 25
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

DICTS = {
    "4x4_50": cv2.aruco.DICT_4X4_50,
    "5x5_100": cv2.aruco.DICT_5X5_100,
    "5x5_250": cv2.aruco.DICT_5X5_250,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel-w", type=int, default=1872, help="Panel width in px.")
    ap.add_argument("--panel-h", type=int, default=1404, help="Panel height in px.")
    ap.add_argument("--ppi", type=float, default=227.0, help="Panel pixel density (px per inch).")
    ap.add_argument("--squares-x", type=int, default=8, help="ChArUco squares along X.")
    ap.add_argument("--squares-y", type=int, default=6, help="ChArUco squares along Y.")
    ap.add_argument("--square-mm", type=float, default=25.0, help="Nominal square edge in mm.")
    ap.add_argument("--marker-ratio", type=float, default=0.75, help="marker_len / square_len.")
    ap.add_argument("--dict", choices=list(DICTS), default="5x5_100", help="ArUco dictionary.")
    ap.add_argument("--out", type=str, default="handeye/calibration_d405/charuco_boox_go_103.png", help="Output PNG path.")
    args = ap.parse_args()

    px_per_mm = args.ppi / 25.4
    square_px = int(round(args.square_mm * px_per_mm))
    board_w = args.squares_x * square_px
    board_h = args.squares_y * square_px
    if board_w > args.panel_w or board_h > args.panel_h:
        raise SystemExit(
            f"Board {board_w}x{board_h}px exceeds panel {args.panel_w}x{args.panel_h}px. "
            "Reduce --square-mm or square counts."
        )

    square_len_m = args.square_mm / 1000.0
    marker_len_m = square_len_m * args.marker_ratio
    dictionary = cv2.aruco.getPredefinedDictionary(DICTS[args.dict])
    board = cv2.aruco.CharucoBoard((args.squares_x, args.squares_y), square_len_m, marker_len_m, dictionary)
    board_img = board.generateImage((board_w, board_h), marginSize=0, borderBits=1)

    # Center the board on a white, panel-sized canvas so full-screen display is 1:1.
    canvas = np.full((args.panel_h, args.panel_w), 255, dtype=np.uint8)
    y0 = (args.panel_h - board_h) // 2
    x0 = (args.panel_w - board_w) // 2
    canvas[y0:y0 + board_h, x0:x0 + board_w] = board_img

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)

    params = {
        "dictionary": args.dict,
        "squares_x": args.squares_x,
        "squares_y": args.squares_y,
        "square_length_m_nominal": round(square_len_m, 6),
        "marker_length_m_nominal": round(marker_len_m, 6),
        "square_px": square_px,
        "panel_px": [args.panel_w, args.panel_h],
        "ppi": args.ppi,
        "note": "MEASURE the displayed square with a ruler and overwrite square_length_m before calibrating.",
    }
    params_path = out_path.with_suffix(".json")
    params_path.write_text(json.dumps(params, indent=2))

    print(f"Wrote {out_path}  ({args.panel_w}x{args.panel_h}px)")
    print(f"Wrote {params_path}")
    print(f"Board: {args.squares_x}x{args.squares_y} squares, nominal square {args.square_mm} mm "
          f"({square_px}px), marker {marker_len_m*1000:.1f} mm, dict {args.dict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
