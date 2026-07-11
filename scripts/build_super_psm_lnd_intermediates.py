#!/usr/bin/env python3
"""
Build PSM LND forward kinematics intermediates from scratch.

Inputs:
  data/LND.json          — Modified DH params + point features + skeleton
  data/handeye.yaml       — PSM base in raw camera (rvec/tvec in mm)
  data/camera_calibration.yaml — stereo calibration
  data/super/grasp5_native/joints.json — 5458 joint states
  data/super/grasp5_native/calib_rectified.json — rectified calibration

Outputs:
  data/super/grasp5_offline_demo/instruments/
    psm1_lnd_model.json   — FK model + transforms
    psm1_lnd_motion.json  — per-frame keypoints + link transforms
    psm1_lnd_generation_report.json
"""

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]

JOINT_NAMES_7 = [
    "outer_yaw", "outer_pitch", "outer_insertion", "outer_roll",
    "outer_wrist_pitch", "outer_wrist_yaw", "jaw",
]


def parse_args():
    p = argparse.ArgumentParser(description="Build PSM LND FK intermediates")
    p.add_argument("--lnd", type=Path, default=REPO / "data/LND.json")
    p.add_argument("--handeye", type=Path, default=REPO / "data/handeye.yaml")
    p.add_argument("--calib", type=Path, default=REPO / "data/camera_calibration.yaml")
    p.add_argument("--calib-rect", type=Path, default=REPO / "data/super/grasp5_native/calib_rectified.json")
    p.add_argument("--joints", type=Path, default=REPO / "data/super/grasp5_native/joints.json")
    p.add_argument("--out-dir", type=Path, default=REPO / "data/super/grasp5_offline_demo/instruments")
    p.add_argument("--debug-frames", type=int, nargs="+", default=[0])
    return p.parse_args()


# ── file parsing ──────────────────────────────────────────────────────────────

def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def parse_lnd_json(path: Path) -> dict:
    """Parse LND JSON that may contain // comments (non-standard)."""
    text = path.read_text()
    text = re.sub(r"//.*", "", text)
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return json.loads(text)


def parse_handeye_yaml(path: Path) -> dict[str, np.ndarray]:
    """Parse handeye.yaml into numpy arrays."""
    text = path.read_text()
    out = {}
    for match in re.finditer(r"(\w+):\s*\[(.*?)\]", text, flags=re.DOTALL):
        values = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", match.group(2))]
        out[match.group(1)] = np.asarray(values, dtype=np.float64)
    return out


def read_opencv_yaml_matrix(path: Path, key: str) -> np.ndarray:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    m = fs.getNode(key).mat()
    fs.release()
    return m


def read_opencv_yaml_vec(path: Path, key: str) -> np.ndarray:
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    node = fs.getNode(key)
    values = np.array([node.at(i).real() for i in range(node.size())], dtype=np.float64)
    fs.release()
    return values


# ── transforms ────────────────────────────────────────────────────────────────

def rotx(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]], dtype=np.float64)


def rotz(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)


def transx(x: float) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[0, 3] = x
    return t


def transz(z: float) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[2, 3] = z
    return t


def modified_dh(alpha: float, a: float, theta: float, d: float) -> np.ndarray:
    """Craig modified DH: Rx(alpha) @ Tx(A) @ Rz(theta) @ Tz(D)."""
    return rotx(alpha) @ transx(a) @ rotz(theta) @ transz(d)


def rpy_to_mat(r: float, p: float, y: float) -> np.ndarray:
    """RPY rotation → 3×3 matrix (fixed-axis: Rz(y) @ Ry(p) @ Rx(r))."""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


# ── LND FK ────────────────────────────────────────────────────────────────────

def lnd_forward_kinematics(lnd: dict, q: np.ndarray) -> dict[int, np.ndarray]:
    """
    Compute LND link transforms via Modified DH.
    q[0:6] are the 6 DH joint values.
    q[6] (jaw) controls gripper links 7 and 8 via ±0.5*jaw rotation.
    Returns: {link_index: 4×4 transform}
    """
    transforms = {0: np.eye(4, dtype=np.float64)}
    t = np.eye(4, dtype=np.float64)

    for i, dh in enumerate(lnd["DH_params"], start=1):
        theta0 = float(dh.get("theta", 0.0))
        d0 = float(dh.get("D", 0.0))
        offset = float(dh.get("offset", 0.0))
        if dh["type"] == "revolute":
            theta = theta0 + float(q[i - 1]) + offset
            d = d0
        elif dh["type"] == "prismatic":
            theta = theta0
            d = d0 + float(q[i - 1]) + offset
        else:
            raise ValueError(f"Unknown joint type: {dh['type']}")
        t = t @ modified_dh(float(dh.get("alpha", 0.0)), float(dh.get("A", 0.0)), theta, d)
        transforms[i] = t.copy()

    # Gripper (jaw): links 7 and 8 rotate about Z by ±0.5 * jaw
    jaw = float(np.clip(q[6], -1.2, 1.6)) if len(q) > 6 else 0.0
    transforms[7] = transforms[6] @ rotz(0.5 * jaw)
    transforms[8] = transforms[6] @ rotz(-0.5 * jaw)
    return transforms


def lnd_points_and_lines(lnd: dict, link_t: dict[int, np.ndarray]) -> tuple[dict, list]:
    """Extract named keypoints and skeleton lines from LND FK result.
    LND feature positions are stored in meters; adjacent comments show their
    millimeter equivalents for readability.
    """
    points = {}
    lines = []
    for item in lnd.get("point_features", []):
        pos_m = np.asarray(item["position"], dtype=np.float64)
        points[item["name"]] = (link_t[int(item["link"])] @ np.r_[pos_m, 1.0])[:3]

    for i, item in enumerate(lnd.get("skeleton_structure", [])):
        p1_m = np.asarray(item["position1"], dtype=np.float64)
        p2_m = np.asarray(item["position2"], dtype=np.float64)
        a, b = f"skeleton_{i}_a", f"skeleton_{i}_b"
        points[a] = (link_t[int(item["link1"])] @ np.r_[p1_m, 1.0])[:3]
        points[b] = (link_t[int(item["link2"])] @ np.r_[p2_m, 1.0])[:3]
        lines.append((a, b))

    for item in lnd.get("shaft_features", []):
        link = int(item["link"])
        pos_m = np.asarray(item["position"], dtype=np.float64)
        direction = np.asarray(item["direction"], dtype=np.float64)
        direction = link_t[link][:3, :3] @ direction
        direction /= np.linalg.norm(direction) + 1e-12
        origin = (link_t[link] @ np.r_[pos_m, 1.0])[:3]
        a, b = f"{item['name']}_near", f"{item['name']}_far"
        points[a] = origin - 0.08 * direction
        points[b] = origin + 0.08 * direction
        lines.append((a, b))

    return points, lines


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # 1. Load input data
    lnd = parse_lnd_json(args.lnd)
    handeye = parse_handeye_yaml(args.handeye)
    joints_data = load_json(args.joints)
    calib_rect = load_json(args.calib_rect)

    # 2. Build handeye transform (rvec/tvec → 4×4)
    rvec = handeye["PSM1_rvec"].reshape(3, 1)
    tvec_mm = handeye["PSM1_tvec"].reshape(3, 1)
    R_he, _ = cv2.Rodrigues(rvec)
    T_raw_psm = np.eye(4, dtype=np.float64)
    T_raw_psm[:3, :3] = R_he
    T_raw_psm[:3, 3] = (tvec_mm * 0.001).ravel()  # mm → m

    # 3. Rectification: R1 maps raw camera → rectified left camera
    R1 = np.asarray(calib_rect["R1"], dtype=np.float64).reshape(3, 3)
    T_rect_raw = np.eye(4, dtype=np.float64)
    T_rect_raw[:3, :3] = R1

    # Composite: rectified_left_camera_from_psm_base
    T_rect_psm = T_rect_raw @ T_raw_psm

    # 4. Camera intrinsics
    K_rect_full = np.asarray(calib_rect.get("K_rectified", calib_rect["P1"][:3]), dtype=np.float64)
    K_left_half = np.asarray(calib_rect.get("K_left_rect", K_rect_full), dtype=np.float64)

    # 5. Process all frames
    states = joints_data["states"]
    motion_states = []
    lines = []

    for frame_idx, state in enumerate(states):
        q7 = np.asarray(state["q"], dtype=np.float64)
        link_base = lnd_forward_kinematics(lnd, q7)
        pts_base, frame_lines = lnd_points_and_lines(lnd, link_base)
        lines = frame_lines  # same for all frames

        # Transform keypoints through coordinate chain
        pts_raw_camera = {}
        pts_rectified_camera = {}
        for name, p in pts_base.items():
            p_rect = (T_rect_psm @ np.r_[p, 1.0])[:3]
            pts_rectified_camera[name] = p_rect.tolist()

        motion_states.append({
            "frame_index": frame_idx,
            "timestamp": float(joints_data["states_timestamps"][frame_idx]),
            "q": q7.tolist(),
            "keypoints_rectified_camera": pts_rectified_camera,
            "link_transforms_psm_base": {str(k): v.tolist() for k, v in link_base.items()},
        })

    # 6. Write model
    model = {
        "name": "psm1_lnd",
        "generated_by": "build_super_psm_lnd_intermediates.py",
        "dh_convention": "Craig modified DH: Rx(alpha) @ Tx(A) @ Rz(theta) @ Tz(D)",
        "joint_names": JOINT_NAMES_7,
        "num_frames": len(motion_states),
        "dh_params": lnd.get("DH_params", []),
        "point_features": lnd.get("point_features", []),
        "skeleton_structure": lnd.get("skeleton_structure", []),
        "shaft_features": lnd.get("shaft_features", []),
        "T_raw_camera_psm_base": T_raw_psm.tolist(),
        "T_rectified_camera_raw": T_rect_raw.tolist(),
        "T_rectified_camera_psm_base": T_rect_psm.tolist(),
        "K_rectified_full": K_rect_full.tolist(),
        "K_left_rect_half": K_left_half.tolist(),
        "R1_raw_to_rectified_left": R1.tolist(),
    }
    save_json(args.out_dir / "psm1_lnd_model.json", model)

    motion = {
        "name": "psm1_lnd_motion",
        "model_path": "psm1_lnd_model.json",
        "num_frames": len(motion_states),
        "joint_names": JOINT_NAMES_7,
        "skeleton_line_names": [[a, b] for a, b in lines],
        "states": motion_states,
    }
    save_json(args.out_dir / "psm1_lnd_motion.json", motion)

    # 7. Quick validation: project frame 0 keypoints
    print(f"Generated {len(motion_states)} frames")
    s0 = motion_states[0]
    pts = {k: np.asarray(v) for k, v in s0["keypoints_rectified_camera"].items()}
    w, h = 1920, 1080
    in_img = 0
    for name, p in pts.items():
        if p[2] > 1e-9:
            uv = K_rect_full[:2, :3] @ (p / p[2])
            if 0 <= uv[0] < w and 0 <= uv[1] < h:
                in_img += 1
    print(f"Frame 0: {in_img}/{len(pts)} keypoints in rectified left image")

    # Report
    report = {
        "num_frames": len(motion_states),
        "frame0_keypoints_in_image": f"{in_img}/{len(pts)}",
        "T_raw_psm_translation_m": T_raw_psm[:3, 3].tolist(),
        "T_rect_psm_translation_m": T_rect_psm[:3, 3].tolist(),
    }
    save_json(args.out_dir / "psm1_lnd_generation_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
