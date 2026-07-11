#!/usr/bin/env python3
"""
Fit PSM URDF fixed-base origin to LND keypoints in rectified camera frame.

1. Load URDF, temporarily set fixed base to identity → compute FK → get link origins
2. Load LND keypoints from psm1_lnd_motion.json frame 0
3. Establish correspondences between URDF links and LND reference points
4. Solve rigid transform (Umeyama) → convert to rpy/xyz
5. Write corrected origin into psm.urdf
6. Validate: re-FK with new origin and report residuals
"""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]

# Known mapping from LND link index to URDF link name
LND_LINK_TO_URDF = {
    0: "PSM1_psm_base_link",
    1: "PSM1_outer_yaw_link",
    2: "PSM1_outer_pitch_link",
    3: "PSM1_tool_main_link",       # insertion link
    4: "PSM1_tool_wrist_link",
    5: "PSM1_tool_wrist_sca_link",
    6: "PSM1_tool_wrist_sca_shaft_link",
}

# Correspondence: (URDF link name, LND link index, weight)
# Maps URDF link origins to LND link origins (from rectified camera frame FK).
# LND link indices: 0=base, 1=yaw, 2=pitch, 3=insertion, 4=wrist/roll, 5=wrist_pitch, 6=wrist_yaw
CORRESPONDENCES = [
    # (urdf_link, lnd_link_index, weight)
    ("PSM1_psm_base_link",          0,  3.0),   # base
    ("PSM1_outer_pitch_link",       2,  3.0),   # pitch link
    ("PSM1_tool_main_link",         3,  3.0),   # insertion link
    ("PSM1_tool_wrist_link",        4,  2.0),   # roll/wrist link
    ("PSM1_tool_wrist_sca_shaft_link", 6, 2.0),   # distal shaft
    ("PSM1_tool_tip_link",          6,  0.5),   # tip (use link 6 as rough target)
]


def parse_args():
    p = argparse.ArgumentParser(description="Fit PSM URDF base origin to LND keypoints")
    p.add_argument("--urdf", type=Path, default=REPO / "data/super/psm_robot/psm.urdf")
    p.add_argument("--mimic-map", type=Path, default=REPO / "data/super/psm_robot/psm_mimic_map.json")
    p.add_argument("--lnd-motion", type=Path,
                   default=REPO / "data/super/grasp5_offline_demo/instruments/psm1_lnd_motion.json")
    p.add_argument("--joints", type=Path, default=REPO / "data/super/grasp5_native/joints.json")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="Print result without modifying URDF")
    return p.parse_args()


# ── file I/O ──────────────────────────────────────────────────────────────────

def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── URDF FK (lightweight, no external deps beyond stdlib) ──────────────────────

def parse_urdf_tree(urdf_path: Path):
    """Parse URDF into link and joint dictionaries."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    links = {}
    joints = {}
    children = {}
    for link in root.findall("link"):
        links[link.attrib["name"]] = link
    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        jtype = joint.attrib["type"]
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        origin = joint.find("origin")
        if origin is not None:
            xyz = [float(v) for v in origin.attrib.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in origin.attrib.get("rpy", "0 0 0").split()]
        else:
            xyz, rpy = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        axis_elem = joint.find("axis")
        axis = [float(v) for v in axis_elem.attrib.get("xyz", "0 0 1").split()] if axis_elem is not None else [0.0, 0.0, 1.0]
        joints[name] = {
            "type": jtype, "parent": parent, "child": child,
            "xyz": xyz, "rpy": rpy, "axis": axis,
        }
        children.setdefault(parent, []).append(name)
    return links, joints, children


def origin_to_mat(xyz: list[float], rpy: list[float]) -> np.ndarray:
    """URDF origin (xyz, rpy) → 4×4 homogeneous transform."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def urdf_fk(joints: dict, children: dict, q_by_joint: dict) -> dict[str, np.ndarray]:
    """
    Forward kinematics for URDF tree.
    q_by_joint: {joint_name: joint_value} for all joints.
    Returns: {link_name: 4×4 transform from world}
    """
    link_tf = {"world": np.eye(4, dtype=np.float64)}
    q_default = 0.0

    # BFS traversal from world
    queue = ["world"]
    while queue:
        parent_link = queue.pop(0)
        for jname in children.get(parent_link, []):
            jinfo = joints[jname]
            child_link = jinfo["child"]
            q_val = q_by_joint.get(jname, q_default)
            T_origin = origin_to_mat(jinfo["xyz"], jinfo["rpy"])

            if jinfo["type"] in ("revolute", "continuous"):
                axis = np.asarray(jinfo["axis"], dtype=np.float64)
                axis /= np.linalg.norm(axis) + 1e-12
                T_joint = np.eye(4, dtype=np.float64)
                T_joint[:3, :3] = R.from_rotvec(axis * q_val).as_matrix()
            elif jinfo["type"] == "prismatic":
                axis = np.asarray(jinfo["axis"], dtype=np.float64)
                axis /= np.linalg.norm(axis) + 1e-12
                T_joint = np.eye(4, dtype=np.float64)
                T_joint[:3, 3] = axis * q_val
            elif jinfo["type"] == "fixed":
                T_joint = np.eye(4, dtype=np.float64)
            else:
                T_joint = np.eye(4, dtype=np.float64)

            link_tf[child_link] = link_tf[parent_link] @ T_origin @ T_joint
            queue.append(child_link)
    return link_tf


def expand_q7_to_full(q7: list[float], mimic_map: dict) -> dict[str, float]:
    """Expand 7 joint values to full URDF joint dict using mimic map."""
    input_names = mimic_map["input_joint_names"]
    input_to_urdf = mimic_map["input_to_urdf_joint"]
    q_by_joint = {}
    for iname, val in zip(input_names, q7):
        q_by_joint[input_to_urdf[iname]] = float(val)
    for jname, spec in mimic_map.get("mimic", {}).items():
        src = spec["source"]
        q_by_joint[jname] = q_by_joint[src] * float(spec.get("multiplier", 1.0)) + float(spec.get("offset", 0.0))
    return q_by_joint


# ── rigid fitting ─────────────────────────────────────────────────────────────

def umeyama(A: np.ndarray, B: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Weighted rigid transform: find (R, t) minimizing sum(w_i * ||R*A_i + t - B_i||^2).
    Returns (R_3x3, t_3).
    """
    if weights is None:
        weights = np.ones(len(A), dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    w_sum = w.sum()
    mean_A = (w[:, None] * A).sum(axis=0) / w_sum
    mean_B = (w[:, None] * B).sum(axis=0) / w_sum
    Ac = A - mean_A
    Bc = B - mean_B
    H = (Ac * w[:, None]).T @ Bc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mean_B - R @ mean_A
    return R, t


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # 1. Load data
    joints_data = load_json(args.joints)
    lnd_motion = load_json(args.lnd_motion)
    mimic_map = load_json(args.mimic_map)

    q7 = joints_data["states"][args.frame]["q"]
    lnd_frame = lnd_motion["states"][args.frame]
    lnd_pts_rectified = {k: np.asarray(v, dtype=np.float64)
                         for k, v in lnd_frame["keypoints_rectified_camera"].items()}

    # 2. Parse URDF and compute FK starting from PSM base (skip world→base fixed joint).
    #    This gives URDF link origins in a PSM-base-local frame (identity base).
    links, joints, children = parse_urdf_tree(args.urdf)
    q_full = expand_q7_to_full(q7, mimic_map)

    # FK from PSM base link instead of world (effectively identity base)
    base_link = "PSM1_psm_base_link"
    link_tf = {base_link: np.eye(4, dtype=np.float64)}
    queue = [base_link]
    while queue:
        parent = queue.pop(0)
        for jname in children.get(parent, []):
            jinfo = joints[jname]
            child = jinfo["child"]
            q_val = q_full.get(jname, 0.0)
            T_origin = origin_to_mat(jinfo["xyz"], jinfo["rpy"])
            if jinfo["type"] in ("revolute", "continuous"):
                axis = np.asarray(jinfo["axis"], dtype=np.float64)
                axis /= np.linalg.norm(axis) + 1e-12
                T_joint = np.eye(4, dtype=np.float64)
                T_joint[:3, :3] = R.from_rotvec(axis * q_val).as_matrix()
            elif jinfo["type"] == "prismatic":
                axis = np.asarray(jinfo["axis"], dtype=np.float64)
                axis /= np.linalg.norm(axis) + 1e-12
                T_joint = np.eye(4, dtype=np.float64)
                T_joint[:3, 3] = axis * q_val
            else:
                T_joint = np.eye(4, dtype=np.float64)
            link_tf[child] = link_tf[parent] @ T_origin @ T_joint
            queue.append(child)

    # 4. Build correspondences:
    #    URDF point = link origin in URDF-local frame (identity base)
    #    LND point  = LND link origin in rectified camera frame
    lnd_model = load_json(REPO / "data/super/grasp5_offline_demo/instruments/psm1_lnd_model.json")
    T_rect_psm_lnd = np.asarray(lnd_model["T_rectified_camera_psm_base"], dtype=np.float64)

    lnd_link_tf_psm_base = {}  # LND link transforms in PSM base frame (from motion)
    for k, v in lnd_frame.get("link_transforms_psm_base", {}).items():
        lnd_link_tf_psm_base[int(k)] = np.asarray(v, dtype=np.float64)

    urdf_pts = []
    lnd_pts = []
    weights = []

    for urdf_link, lnd_link_idx, weight in CORRESPONDENCES:
        if weight <= 0:
            continue
        if urdf_link not in link_tf:
            print(f"  WARNING: URDF link {urdf_link} not found")
            continue
        if lnd_link_idx not in lnd_link_tf_psm_base:
            print(f"  WARNING: LND link {lnd_link_idx} not found")
            continue

        # URDF link origin (in URDF-local = PSM base frame with identity fixed joint)
        p_urdf = link_tf[urdf_link][:3, 3].copy()

        # LND link origin in rectified camera frame
        p_lnd_psm = lnd_link_tf_psm_base[lnd_link_idx][:3, 3]
        p_lnd = (T_rect_psm_lnd @ np.r_[p_lnd_psm, 1.0])[:3]

        urdf_pts.append(p_urdf)
        lnd_pts.append(p_lnd)
        weights.append(weight)

    urdf_pts = np.array(urdf_pts, dtype=np.float64)
    lnd_pts = np.array(lnd_pts, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)

    print(f"Correspondences: {len(urdf_pts)} points")
    for i, (up, lp, w) in enumerate(zip(urdf_pts, lnd_pts, weights)):
        print(f"  [{i}] w={w:.1f} URDF({up[0]:.4f},{up[1]:.4f},{up[2]:.4f}) → LND({lp[0]:.4f},{lp[1]:.4f},{lp[2]:.4f})")

    # 5. Solve rigid transform: LND = R @ URDF + t
    R_fit, t_fit = umeyama(urdf_pts, lnd_pts, weights)

    T_fit = np.eye(4, dtype=np.float64)
    T_fit[:3, :3] = R_fit
    T_fit[:3, 3] = t_fit

    # 6. Convert to rpy + xyz
    rpy = R.from_matrix(R_fit).as_euler("xyz")
    xyz = t_fit

    # 7. Residuals
    residuals = []
    for i, (up, lp, w) in enumerate(zip(urdf_pts, lnd_pts, weights)):
        pred = R_fit @ up + t_fit
        err = np.linalg.norm(pred - lp)
        residuals.append({"urdf_link": CORRESPONDENCES[i][0],
                          "lnd_link": CORRESPONDENCES[i][1],
                          "weight": float(w), "residual_m": float(err),
                          "predicted": pred.tolist(), "target": lp.tolist()})

    mean_res = np.mean([r["residual_m"] for r in residuals])
    max_res = np.max([r["residual_m"] for r in residuals])
    print(f"\nFit result: rpy={[f'{v:.6f}' for v in rpy]} xyz={[f'{v:.6f}' for v in xyz]}")
    print(f"Residuals: mean={mean_res*1000:.2f}mm max={max_res*1000:.2f}mm")
    for r in residuals:
        print(f"  {r['urdf_link']:35s} → LND link {r['lnd_link']}: {r['residual_m']*1000:.2f}mm")

    # 8. Write back to URDF (or dry-run)
    if not args.dry_run:
        urdf_tree = ET.parse(args.urdf)
        urdf_root = urdf_tree.getroot()
        fixed_joint = urdf_root.find("./joint[@name='fixed']")
        origin_elem = fixed_joint.find("origin")
        if origin_elem is None:
            origin_elem = ET.SubElement(fixed_joint, "origin")
        origin_elem.set("rpy", f"{rpy[0]:.9f} {rpy[1]:.9f} {rpy[2]:.9f}")
        origin_elem.set("xyz", f"{xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f}")
        ET.indent(urdf_tree, space="  ")
        urdf_tree.write(args.urdf, encoding="utf-8", xml_declaration=True)
        print(f"\nWritten to {args.urdf}")
    else:
        print("\nDry run — URDF not modified.")

    # 9. Save report
    report = {
        "frame": args.frame,
        "num_correspondences": len(urdf_pts),
        "mean_residual_m": float(mean_res),
        "max_residual_m": float(max_res),
        "fitted_rpy": rpy.tolist(),
        "fitted_xyz": xyz.tolist(),
        "T_rectified_camera_psm_base": T_fit.tolist(),
        "residuals": residuals,
    }
    save_json(REPO / "data/super/psm_robot/psm_base_correction_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "residuals"}, indent=2))


if __name__ == "__main__":
    main()
