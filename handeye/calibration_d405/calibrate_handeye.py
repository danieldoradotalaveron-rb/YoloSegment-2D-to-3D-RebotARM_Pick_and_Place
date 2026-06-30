"""Solve eye-in-hand calibration from a captured session.

Loads a session written by capture_handeye.py, detects the ChArUco board in each
saved image to get `cam_T_target`, pairs it with the recorded `base_T_ee`, and
runs cv2.calibrateHandEye to recover `ee_T_cam` (the camera pose in the
end-effector frame).

OpenCV convention (eye-in-hand):
    R/t_gripper2base = base_T_ee   (pose of ee in base)        <- from TF
    R/t_target2cam   = cam_T_target (pose of target in camera) <- from solvePnP
    output           = ee_T_cam     (pose of camera in ee)     <- what we want

Validation: the board is physically fixed, so
    base_T_target = base_T_ee @ ee_T_cam @ cam_T_target
must be (near) constant across all poses. We report the spread of that estimate;
small translation/rotation std => good calibration.

Run:
    .venv/bin/python handeye/calibration_d405/calibrate_handeye.py handeye/calibration_d405/captures/session_XXXX
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

HANDEYE_DIR = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(HANDEYE_DIR))

from charuco_utils import detect, estimate_pose, intrinsics_to_K, load_board  # noqa: E402
from transforms import R_to_quat, make_T, rotation_angle_deg  # noqa: E402

METHODS = {
    "TSAI": cv2.CALIB_HAND_EYE_TSAI,
    "PARK": cv2.CALIB_HAND_EYE_PARK,
    "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
    "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
    "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", help="Path to a capture session dir (containing session.json) or the json itself.")
    ap.add_argument("--min-corners", type=int, default=8)
    ap.add_argument("--out", default="", help="Where to write ee_T_cam.json (default: alongside the session).")
    return ap.parse_args()


def load_session(path: Path) -> tuple[dict, Path]:
    if path.is_dir():
        path = path / "session.json"
    return json.loads(path.read_text()), path.parent


def resolve_image_path(stored: str, session_dir: Path) -> Path:
    """Use the stored path if it exists, else fall back to the same basename inside
    the session dir. Makes calibration robust to a moved/renamed session folder."""
    p = Path(stored)
    if p.exists():
        return p
    return session_dir / p.name


def resolve_board_path(stored: str, session_dir: Path) -> str:
    """Stored board_json path may be stale if the folder moved; fall back to a board
    next to the session, then to its basename in the session dir."""
    p = Path(stored)
    if p.exists():
        return str(p)
    for candidate in (session_dir / p.name, session_dir.parent.parent / p.name):
        if candidate.exists():
            return str(candidate)
    return stored


def main() -> int:
    args = parse_args()
    session, session_dir = load_session(Path(args.session))
    spec = load_board(resolve_board_path(session["board_json"], session_dir))
    K, dist = intrinsics_to_K(session["intrinsics"])

    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    base_T_ee_list, cam_T_target_list = [], []
    used = 0
    for s in session["samples"]:
        img_path = resolve_image_path(s["image"], session_dir)
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[skip] cannot read {img_path}")
            continue
        det = detect(spec, img)
        pose = estimate_pose(spec, det, K, dist, min_corners=args.min_corners)
        if pose is None:
            print(f"[skip] sample {s['index']}: weak board view (corners={det.n_corners})")
            continue
        rvec, tvec = pose
        R_tc, _ = cv2.Rodrigues(rvec)
        cam_T_target = make_T(R_tc, tvec)

        base_T_ee = np.asarray(s["base_T_ee"], dtype=np.float64)
        R_g2b.append(base_T_ee[:3, :3])
        t_g2b.append(base_T_ee[:3, 3])
        R_t2c.append(R_tc)
        t_t2c.append(tvec)
        base_T_ee_list.append(base_T_ee)
        cam_T_target_list.append(cam_T_target)
        used += 1

    print(f"[data] {used}/{len(session['samples'])} samples usable")
    if used < 3:
        print("[error] need at least 3 usable samples (ideally 12+).")
        return 1

    best = None
    for name, flag in METHODS.items():
        try:
            R_c2g, t_c2g = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=flag)
        except cv2.error as exc:
            print(f"[{name}] failed: {exc}")
            continue
        ee_T_cam = make_T(R_c2g, t_c2g)
        trans_std, rot_std = validate(ee_T_cam, base_T_ee_list, cam_T_target_list)
        score = trans_std * 1000.0 + rot_std  # mm + deg, rough combined residual
        print(
            f"[{name:11s}] cam pos in ee = {np.round(ee_T_cam[:3, 3]*1000, 1).tolist()} mm | "
            f"base_T_target spread: trans={trans_std*1000:.2f} mm, rot={rot_std:.3f} deg"
        )
        if best is None or score < best["score"]:
            best = {"name": name, "ee_T_cam": ee_T_cam, "trans_std": trans_std, "rot_std": rot_std, "score": score}

    if best is None:
        print("[error] all methods failed")
        return 1

    ee_T_cam = best["ee_T_cam"]
    print("\n=== BEST: " + best["name"] + " ===")
    print("ee_T_cam (4x4):")
    print(np.array2string(ee_T_cam, precision=5, suppress_small=True))
    print(
        f"camera position in end_link: {np.round(ee_T_cam[:3, 3]*1000, 2).tolist()} mm\n"
        f"validation: board pos spread {best['trans_std']*1000:.2f} mm, "
        f"rot spread {best['rot_std']:.3f} deg (lower = better)"
    )

    quat = R_to_quat(ee_T_cam[:3, :3])
    out = {
        "frame_parent": session["ee_frame"],
        "frame_child": "camera",
        "method": best["name"],
        "ee_T_cam": ee_T_cam.tolist(),
        "translation_m": ee_T_cam[:3, 3].tolist(),
        "quaternion_xyzw": quat.tolist(),
        "validation": {"board_trans_std_mm": best["trans_std"] * 1000, "board_rot_std_deg": best["rot_std"]},
        "n_samples": used,
    }
    out_path = Path(args.out) if args.out else session_dir / "ee_T_cam.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[written] {out_path}")
    return 0


def validate(
    ee_T_cam: np.ndarray, base_T_ee_list: list[np.ndarray], cam_T_target_list: list[np.ndarray]
) -> tuple[float, float]:
    """Spread of base_T_target across poses: translation std (m) and mean pairwise
    rotation deviation (deg). The board is fixed, so a good ee_T_cam makes these ~0."""
    targets = [bte @ ee_T_cam @ ctt for bte, ctt in zip(base_T_ee_list, cam_T_target_list)]
    trans = np.array([T[:3, 3] for T in targets])
    trans_std = float(np.linalg.norm(trans.std(axis=0)))

    R_mean = np.mean([T[:3, :3] for T in targets], axis=0)
    U, _, Vt = np.linalg.svd(R_mean)
    R_ref = U @ Vt
    rot_devs = [rotation_angle_deg(T[:3, :3].T @ R_ref) for T in targets]
    return trans_std, float(np.mean(rot_devs))


if __name__ == "__main__":
    raise SystemExit(main())
