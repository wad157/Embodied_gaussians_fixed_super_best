#!/usr/bin/env python3
"""
Validate PSM URDF alignment by projecting mesh samples + LND keypoints
onto rectified stereo images.

Usage:
  python scripts/validate_psm_projection.py --frame 0
  python scripts/validate_psm_projection.py --frame 0 --cameras stereo_left,stereo_right
"""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]

# Lightweight URDF FK (duplicated from fit script for independence)
def origin_to_mat(xyz, rpy):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T

def urdf_fk(joints, children, q_by_joint):
    link_tf = {"world": np.eye(4, dtype=np.float64)}
    queue = ["world"]
    while queue:
        parent = queue.pop(0)
        for jname in children.get(parent, []):
            jinfo = joints[jname]
            child = jinfo["child"]
            q = q_by_joint.get(jname, 0.0)
            axis = np.asarray(jinfo["axis"], dtype=np.float64)
            n = np.linalg.norm(axis) + 1e-12
            axis = axis / n
            if jinfo["type"] in ("revolute", "continuous"):
                Tj = np.eye(4); Tj[:3,:3] = R.from_rotvec(axis * q).as_matrix()
            elif jinfo["type"] == "prismatic":
                Tj = np.eye(4); Tj[:3, 3] = axis * q
            else:
                Tj = np.eye(4)
            link_tf[child] = link_tf[parent] @ origin_to_mat(jinfo["xyz"], jinfo["rpy"]) @ Tj
            queue.append(child)
    return link_tf

def parse_urdf(path):
    root = ET.parse(path).getroot()
    joints, children = {}, {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in o.attrib.get("xyz","0 0 0").split()] if o is not None else [0,0,0]
        rpy = [float(v) for v in o.attrib.get("rpy","0 0 0").split()] if o is not None else [0,0,0]
        ax = j.find("axis")
        axis = [float(v) for v in ax.attrib.get("xyz","0 0 1").split()] if ax is not None else [0.0, 0.0, 1.0]
        joints[j.attrib["name"]] = {
            "type": j.attrib["type"],
            "parent": j.find("parent").attrib["link"],
            "child": j.find("child").attrib["link"],
            "xyz": xyz, "rpy": rpy, "axis": axis,
        }
        children.setdefault(joints[j.attrib["name"]]["parent"], []).append(j.attrib["name"])
    return joints, children

def expand_q7(q7, mimic_map):
    q = {}
    for iname, val in zip(mimic_map["input_joint_names"], q7):
        q[mimic_map["input_to_urdf_joint"][iname]] = float(val)
    for jname, spec in mimic_map.get("mimic", {}).items():
        q[jname] = q[spec["source"]] * spec.get("multiplier",1) + spec.get("offset",0)
    return q

def sample_link_points(link_name, link_tf, urdf_root):
    """Sample points on a URDF link's visual/collision mesh. Returns world-frame points."""
    link = urdf_root.find(f"./link[@name='{link_name}']")
    if link is None:
        return np.zeros((0,3))
    T_link = link_tf[link_name]
    pts = []
    for geom in link.findall(".//geometry/mesh"):
        # Simple: sample the bounding box or just return origin
        pass
    # Return origin as single point for each link
    p = T_link[:3, 3]
    return p.reshape(1, 3)

def project(pts, K, w, h):
    """Project 3D points to image plane. Returns uv, depth, valid_mask."""
    if len(pts) == 0:
        return np.zeros((0,2)), np.zeros(0), np.zeros(0, dtype=bool)
    uv = (K[:2,:3] @ (pts / pts[:,2:3]).T).T
    d = pts[:, 2]
    valid = (d > 1e-9) & (uv[:,0] >= 0) & (uv[:,0] < w) & (uv[:,1] >= 0) & (uv[:,1] < h)
    return uv, d, valid

def draw_points(img, uv, color, radius=3, label=""):
    for u, v in uv.astype(int):
        cv2.circle(img, (u, v), radius, color, -1, cv2.LINE_AA)
    if label and len(uv) > 0:
        cv2.putText(img, label, tuple(uv[0].astype(int) + [5, -5]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--urdf", type=Path, default=REPO / "data/super/psm_robot/psm.urdf")
    p.add_argument("--mimic-map", type=Path, default=REPO / "data/super/psm_robot/psm_mimic_map.json")
    p.add_argument("--joints", type=Path, default=REPO / "data/super/grasp5_native/joints.json")
    p.add_argument("--lnd-motion", type=Path, default=REPO / "data/super/grasp5_offline_demo/instruments/psm1_lnd_motion.json")
    p.add_argument("--rgb-dir", type=Path, default=REPO / "data/super/grasp5_native/rgb")
    p.add_argument("--calib-rect", type=Path, default=REPO / "data/super/grasp5_native/calib_rectified.json")
    p.add_argument("--out-dir", type=Path, default=REPO / "data/super/psm_robot/validation")
    p.add_argument("--cameras", default="stereo_left,stereo_right")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    joints_data = json.load(open(args.joints))
    lnd_motion = json.load(open(args.lnd_motion))
    mimic_map = json.load(open(args.mimic_map))
    calib = json.load(open(args.calib_rect))
    K_full = np.asarray(calib.get("K_rectified", calib.get("K_left_rect")), dtype=np.float64).reshape(3,3)
    K_half = np.asarray(calib.get("K_left_rect", K_full), dtype=np.float64).reshape(3,3)

    urdf_root = ET.parse(args.urdf).getroot()
    joints, children = parse_urdf(args.urdf)
    joint_timestamps = np.asarray(joints_data["states_timestamps"], dtype=np.float64)
    baseline_m = float(calib["baseline_m"])

    # Key URDF link origins to visualize
    key_links = [
        ("PSM1_tool_main_link",       (255, 200, 0)),    # yellow
        ("PSM1_tool_wrist_link",      (0, 200, 255)),    # orange
        ("PSM1_tool_wrist_sca_link",  (0, 255, 200)),    # cyan
        ("PSM1_tool_wrist_sca_shaft_link", (200, 0, 255)), # magenta
        ("PSM1_tool_tip_link",        (255, 0, 0)),      # red
    ]

    w, h = 1920, 1080

    for cam_name in args.cameras.split(","):
        cam_name = cam_name.strip()
        side = "left" if "left" in cam_name else "right"
        camera_timestamps = np.asarray(calib[f"{side}_timestamps"], dtype=np.float64)
        if not 0 <= args.frame < len(camera_timestamps):
            raise IndexError(
                f"Video frame {args.frame} outside {side} range "
                f"[0, {len(camera_timestamps) - 1}]"
            )
        camera_timestamp = camera_timestamps[args.frame]
        joint_frame = int(np.argmin(np.abs(joint_timestamps - camera_timestamp)))
        timestamp_error_ms = abs(joint_timestamps[joint_frame] - camera_timestamp) * 1000.0
        q_full = expand_q7(joints_data["states"][joint_frame]["q"], mimic_map)
        link_tf = urdf_fk(joints, children, q_full)
        lnd_frame = lnd_motion["states"][joint_frame]
        lnd_pts = {
            key: np.asarray(value)
            for key, value in lnd_frame["keypoints_rectified_camera"].items()
        }
        X_camera_left = np.eye(4, dtype=np.float64)
        if side == "right":
            X_camera_left[0, 3] = -baseline_m

        rgb_path = args.rgb_dir / f"{args.frame:06d}-{side}.png"
        img = cv2.imread(str(rgb_path))
        if img is None:
            print(f"WARNING: cannot read {rgb_path}")
            continue

        K = K_full  # rectified cameras share same intrinsics
        overlay = img.copy()

        # Draw URDF link origins
        for link_name, color in key_links:
            if link_name in link_tf:
                p_left = link_tf[link_name][:3, 3]
                p = (X_camera_left @ np.r_[p_left, 1.0])[:3].reshape(1, 3)
                uv, d, valid = project(p, K, w, h)
                if valid.any():
                    draw_points(overlay, uv[valid], color, radius=4, label=link_name.split("_")[-1])

        # Draw LND keypoints
        for name, p in lnd_pts.items():
            p = (X_camera_left @ np.r_[p, 1.0])[:3].reshape(1, 3)
            uv, d, valid = project(p, K, w, h)
            color = (0, 255, 0)  # green for LND
            if valid.any():
                draw_points(overlay, uv[valid], color, radius=2)
                cv2.putText(overlay, name, tuple(uv[0].astype(int) + [3, -3]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)

        # Stats
        def visible(point_left):
            point = (X_camera_left @ np.r_[point_left, 1.0])[:3]
            uv = K @ point / point[2]
            return point[2] > 0 and 0 <= uv[0] < w and 0 <= uv[1] < h

        n_lnd_visible = sum(visible(p) for p in lnd_pts.values())
        n_urdf_visible = sum(
            name in link_tf and visible(link_tf[name][:3, 3]) for name, _ in key_links
        )

        cv2.putText(overlay, f"{cam_name} video={args.frame} joint={joint_frame} dt={timestamp_error_ms:.2f}ms LND_vis={n_lnd_visible} URDF_vis={n_urdf_visible}",
                    (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        out_path = args.out_dir / f"frame{args.frame:06d}_{cam_name}_overlay.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"{out_path}: LND_visible={n_lnd_visible} URDF_visible={n_urdf_visible}")

        # Save side-by-side comparison
        side_by_side = np.hstack([img, overlay])
        out_path2 = args.out_dir / f"frame{args.frame:06d}_{cam_name}_compare.png"
        cv2.imwrite(str(out_path2), side_by_side)

    # Also print URDF+LND link position comparison
    print(f"\n=== Link position comparison (rectified camera frame, meters) ===")
    for name, color in key_links:
        if name in link_tf:
            p_urdf = link_tf[name][:3, 3]
            lnd_key = {"PSM1_tool_main_link": "insertion_shaft_near",
                       "PSM1_tool_wrist_link": "roll_front",
                       "PSM1_tool_wrist_sca_shaft_link": "ee_front",
                       "PSM1_tool_tip_link": "grip_mid"}.get(name, "")
            if lnd_key and lnd_key in lnd_pts:
                p_lnd = lnd_pts[lnd_key]
                dist = np.linalg.norm(p_urdf - p_lnd) * 1000
                print(f"  {name:35s} URDF=({p_urdf[0]:.4f},{p_urdf[1]:.4f},{p_urdf[2]:.4f})  "
                      f"LND=({p_lnd[0]:.4f},{p_lnd[1]:.4f},{p_lnd[2]:.4f})  dist={dist:.1f}mm")
            else:
                print(f"  {name:35s} URDF=({p_urdf[0]:.4f},{p_urdf[1]:.4f},{p_urdf[2]:.4f})")


if __name__ == "__main__":
    main()
