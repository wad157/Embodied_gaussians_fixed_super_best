#!/usr/bin/env python3
"""
Export the full SUPER scene as a single PLY for inspection:
  - PSM robot (URDF FK link origins + sampled mesh skeleton)
  - Tissue body (gaussians + particles)
  - Ground body (gaussians)
  - Ground plane (infinite plane sampled grid)
  - Stereo camera frustums

All output points are in the right-handed, z-up table frame (meters).
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
X_OPENCV_CAMERA_FROM_BLENDER_CAMERA = np.diag([1.0, -1.0, -1.0, 1.0])


# ── URDF FK (same as fit/validate scripts) ────────────────────────────────────

def origin_to_mat(xyz, rpy):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def parse_urdf(path):
    root = ET.parse(path).getroot()
    joints, children = {}, {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = [float(v) for v in o.attrib.get("xyz", "0 0 0").split()] if o is not None else [0.0, 0.0, 0.0]
        rpy = [float(v) for v in o.attrib.get("rpy", "0 0 0").split()] if o is not None else [0.0, 0.0, 0.0]
        ax = j.find("axis")
        axis = [float(v) for v in ax.attrib.get("xyz", "0 0 1").split()] if ax is not None else [0.0, 0.0, 1.0]
        joints[j.attrib["name"]] = {
            "type": j.attrib["type"], "parent": j.find("parent").attrib["link"],
            "child": j.find("child").attrib["link"], "xyz": xyz, "rpy": rpy, "axis": axis,
        }
        children.setdefault(joints[j.attrib["name"]]["parent"], []).append(j.attrib["name"])
    return joints, children


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
                Tj = np.eye(4, dtype=np.float64)
                Tj[:3, :3] = R.from_rotvec(axis * q).as_matrix()
            elif jinfo["type"] == "prismatic":
                Tj = np.eye(4, dtype=np.float64)
                Tj[:3, 3] = axis * q
            else:
                Tj = np.eye(4, dtype=np.float64)
            link_tf[child] = link_tf[parent] @ origin_to_mat(jinfo["xyz"], jinfo["rpy"]) @ Tj
            queue.append(child)
    return link_tf


def expand_q7(q7, mimic_map):
    q = {}
    for iname, val in zip(mimic_map["input_joint_names"], q7):
        q[mimic_map["input_to_urdf_joint"][iname]] = float(val)
    for jname, spec in mimic_map.get("mimic", {}).items():
        q[jname] = q[spec["source"]] * float(spec.get("multiplier", 1.0)) + float(spec.get("offset", 0.0))
    return q


# ── PLY writer ────────────────────────────────────────────────────────────────

class PlyBuilder:
    def __init__(self):
        self.verts = []
        self.colors = []

    def add_points(self, pts, color):
        """pts: (N, 3), color: (r, g, b) uint8"""
        if len(pts) == 0:
            return
        self.verts.append(pts)
        self.colors.append(np.tile(np.array(color, dtype=np.uint8), (len(pts), 1)))

    def add_line(self, a, b, color, n_samples=50):
        t = np.linspace(0, 1, n_samples)[:, None]
        pts = a + t * (b - a)
        self.add_points(pts, color)

    def add_sphere(self, center, radius, color, n=20):
        phi = np.linspace(0, np.pi, n)
        theta = np.linspace(0, 2 * np.pi, n)
        phi, theta = np.meshgrid(phi, theta, indexing="ij")
        x = radius * np.sin(phi) * np.cos(theta) + center[0]
        y = radius * np.sin(phi) * np.sin(theta) + center[1]
        z = radius * np.cos(phi) + center[2]
        pts = np.stack([x, y, z], axis=-1).reshape(-1, 3)
        self.add_points(pts, color)

    def write(self, path):
        all_verts = np.vstack(self.verts)
        all_colors = np.vstack(self.colors)
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {len(all_verts)}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            f.write("end_header\n")
            for p, c in zip(all_verts, all_colors):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
        print(f"{path}: {len(all_verts)} vertices")


# ── PSM mesh sampling from URDF ───────────────────────────────────────────────

def sample_psm_links(link_tf, urdf_root, link_names, color, ply):
    """Sample visual mesh vertices for given URDF links."""
    for link in urdf_root.findall("link"):
        name = link.attrib["name"]
        if name not in link_names or name not in link_tf:
            continue
        T = link_tf[name]
        for visual in link.findall("visual"):
            o = visual.find("origin")
            if o is not None:
                xyz = [float(v) for v in o.attrib.get("xyz", "0 0 0").split()]
                rpy = [float(v) for v in o.attrib.get("rpy", "0 0 0").split()]
                T_vis = origin_to_mat(xyz, rpy)
            else:
                T_vis = np.eye(4)
            T_total = T @ T_vis

            geom = visual.find("geometry")
            if geom is None:
                continue
            mesh = geom.find("mesh")
            if mesh is not None:
                fname = mesh.attrib["filename"]
                mesh_path = REPO / "data/super/psm_robot" / fname
                if mesh_path.exists():
                    try:
                        import trimesh
                        m = trimesh.load(mesh_path, process=False)
                        if isinstance(m, trimesh.Trimesh):
                            verts = np.asarray(m.vertices, dtype=np.float64)
                            # Sample if too dense
                            if len(verts) > 2000:
                                idx = np.linspace(0, len(verts) - 1, 2000).astype(int)
                                verts = verts[idx]
                        else:
                            continue
                    except Exception:
                        continue
                    pts = (T_total[:3, :3] @ verts.T + T_total[:3, 3:4]).T
                    ply.add_points(pts, color)
            # Also add link origin as a sphere
            ply.add_sphere(T[:3, 3], 0.002, (255, 255, 255))


def add_camera_frustum(ply, K, width, height, X_WC, color, depth=0.15):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    corners_img = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64
    )
    corners_camera = np.zeros((4, 3), dtype=np.float64)
    for i, (u, v) in enumerate(corners_img):
        corners_camera[i] = [
            (u - cx) * depth / fx,
            (v - cy) * depth / fy,
            depth,
        ]
    origin_world = X_WC[:3, 3]
    corners_world = (
        X_WC[:3, :3] @ corners_camera.T + X_WC[:3, 3:4]
    ).T
    ply.add_sphere(origin_world, 0.003, color)
    for corner in corners_world:
        ply.add_line(origin_world, corner, color)
    for i in range(4):
        ply.add_line(corners_world[i], corners_world[(i + 1) % 4], color)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--urdf", type=Path, default=REPO / "data/super/psm_robot/psm.urdf")
    p.add_argument("--mimic-map", type=Path, default=REPO / "data/super/psm_robot/psm_mimic_map.json")
    p.add_argument(
        "--psm-gaussians",
        type=Path,
        default=REPO / "data/super/psm_robot/psm_surface_gaussians.npz",
    )
    p.add_argument("--joints", type=Path, default=REPO / "data/super/grasp5_native/joints.json")
    p.add_argument("--tissue-body", type=Path, default=REPO / "data/super/grasp5_native/bodies_v5_table/tissue.json")
    p.add_argument("--ground-body", type=Path, default=REPO / "data/super/grasp5_native/bodies_v5_table/ground.json")
    p.add_argument("--ground-plane", type=Path, default=REPO / "data/super/grasp5_native/bodies_v5_table/ground_plane.json")
    p.add_argument("--table-frame", type=Path, default=REPO / "data/super/table_frame.json")
    p.add_argument("--camera-manifest", type=Path, default=REPO / "data/super/grasp5_offline_demo/cameras.json")
    p.add_argument("--depth", type=Path, default=REPO / "data/super/grasp5_native/depth_v2/000000-depth.npy")
    p.add_argument("--rgb", type=Path, default=REPO / "data/super/grasp5_native/rgb/000000-left.png")
    p.add_argument("--calib", type=Path, default=REPO / "data/super/grasp5_native/calib_rectified.json")
    p.add_argument("--out", type=Path, default=REPO / "data/super/scene_overview_table.ply")
    p.add_argument("--sample-depth", type=int, default=16, help="Sample every Nth pixel for depth point cloud")
    args = p.parse_args()

    ply = PlyBuilder()
    table_frame = json.load(open(args.table_frame))
    X_table_camera = np.asarray(table_frame["X_table_camera"], dtype=np.float64)

    # ── 1. PSM Robot ──────────────────────────────────────────────────────────
    joints_data = json.load(open(args.joints))
    mimic_map = json.load(open(args.mimic_map))
    urdf_root = ET.parse(args.urdf).getroot()
    joints, children = parse_urdf(args.urdf)
    q_full = expand_q7(joints_data["states"][args.frame]["q"], mimic_map)
    link_tf = urdf_fk(joints, children, q_full)
    link_tf = {name: X_table_camera @ X for name, X in link_tf.items()}

    # All PSM links that have visual mesh
    psm_links = []
    for link in urdf_root.findall("link"):
        name = link.attrib["name"]
        if name == "world":
            continue
        if link.find("visual/geometry/mesh") is not None:
            psm_links.append(name)
    # Fallback: add links with collision mesh if no visual
    if len(psm_links) == 0:
        for link in urdf_root.findall("link"):
            name = link.attrib["name"]
            if name != "world" and link.find("collision/geometry/mesh") is not None:
                psm_links.append(name)
    print(f"PSM links with mesh: {psm_links}")
    sample_psm_links(link_tf, urdf_root, psm_links, (255, 120, 0), ply)

    # Surface Gaussian centers are stored in their owning URDF link frames.
    if args.psm_gaussians.exists():
        with np.load(args.psm_gaussians, allow_pickle=False) as asset:
            gaussian_means = asset["means"]
            gaussian_link_ids = asset["link_ids"]
            gaussian_link_names = asset["link_names"].tolist()
        for link_id, link_name in enumerate(gaussian_link_names):
            if link_name not in link_tf:
                raise KeyError(f"PSM Gaussian link missing from FK: {link_name}")
            local_means = gaussian_means[gaussian_link_ids == link_id]
            X_WL = link_tf[link_name]
            world_means = (
                X_WL[:3, :3] @ local_means.T + X_WL[:3, 3:4]
            ).T
            ply.add_points(world_means, (0, 255, 255))

    # Link skeleton (origins connected)
    skeleton_chain = [
        "PSM1_psm_base_link", "PSM1_outer_yaw_link", "PSM1_outer_pitch_link",
        "PSM1_tool_main_link", "PSM1_tool_wrist_link", "PSM1_tool_wrist_shaft_link",
        "PSM1_tool_wrist_sca_link", "PSM1_tool_wrist_sca_shaft_link",
    ]
    for i in range(len(skeleton_chain) - 1):
        a_name, b_name = skeleton_chain[i], skeleton_chain[i + 1]
        if a_name in link_tf and b_name in link_tf:
            ply.add_line(link_tf[a_name][:3, 3], link_tf[b_name][:3, 3], (100, 100, 100))

    # ── 2. Tissue body ────────────────────────────────────────────────────────
    tissue = json.load(open(args.tissue_body))
    X_WB = np.array(tissue["X_WB"])
    # gaussians
    gm = np.array(tissue["gaussians"]["means"])
    gm_w = (X_WB[:3, :3] @ gm.T + X_WB[:3, 3:4]).T
    ply.add_points(gm_w, (255, 40, 40))
    # particles
    if tissue.get("particles") and tissue["particles"]["means"]:
        pm = np.array(tissue["particles"]["means"])
        pm_w = (X_WB[:3, :3] @ pm.T + X_WB[:3, 3:4]).T
        ply.add_points(pm_w, (200, 80, 80))

    # ── 3. Ground body ────────────────────────────────────────────────────────
    ground = json.load(open(args.ground_body))
    X_WB_g = np.array(ground["X_WB"])
    gm_g = np.array(ground["gaussians"]["means"])
    gm_g_w = (X_WB_g[:3, :3] @ gm_g.T + X_WB_g[:3, 3:4]).T
    ply.add_points(gm_g_w, (40, 100, 255))

    # ── 4. Ground plane (infinite) ────────────────────────────────────────────
    plane_data = json.load(open(args.ground_plane))
    plane = np.array(plane_data["plane"])  # [a, b, c, d]
    n_pl = plane[:3]
    d_pl = plane[3]
    # Sample a grid of points on the plane around the tissue
    all_tissue = np.vstack([gm_w]) if len(gm_w) > 0 else np.array([[0, 0, 0.1]])
    cx, cy = all_tissue[:, 0].mean(), all_tissue[:, 1].mean()
    half = max(float(np.ptp(all_tissue[:, 0])), float(np.ptp(all_tissue[:, 1])), 0.05)
    grid_s = np.linspace(cx - half, cx + half, 30)
    grid_t = np.linspace(cy - half, cy + half, 30)
    gs, gt = np.meshgrid(grid_s, grid_t, indexing="ij")
    # Solve for z: a*x + b*y + c*z + d = 0 → z = -(a*x + b*y + d)/c
    if abs(n_pl[2]) > 1e-6:
        gz = -(n_pl[0] * gs + n_pl[1] * gt + d_pl) / n_pl[2]
        plane_pts = np.stack([gs.ravel(), gt.ravel(), gz.ravel()], axis=1)
        ply.add_points(plane_pts, (100, 150, 255))

    # ── 5. Depth point cloud (subsampled) ─────────────────────────────────────
    if args.depth.exists():
        depth = np.load(args.depth)
        calib = json.load(open(args.calib))
        K = np.array(calib["K_left_rect"], dtype=np.float64).reshape(3, 3)
        h, w = depth.shape
        s = args.sample_depth
        vs, us = np.mgrid[0:h:s, 0:w:s].reshape(2, -1)
        z = depth[vs, us]
        valid = np.isfinite(z) & (z > 0.01) & (z < 0.5)
        vs, us, z = vs[valid], us[valid], z[valid]
        x = (us - K[0, 2]) * z / K[0, 0]
        y = (vs - K[1, 2]) * z / K[1, 1]
        depth_pts = np.stack([x, y, z], axis=1)
        depth_pts = (
            X_table_camera[:3, :3] @ depth_pts.T
            + X_table_camera[:3, 3:4]
        ).T
        ply.add_points(depth_pts[::4], (200, 200, 200))  # further subsample for size

    # ── 6. Camera frustums ────────────────────────────────────────────────────
    camera_manifest = json.load(open(args.camera_manifest))
    camera_colors = {
        "stereo_left": (255, 255, 0),
        "stereo_right": (0, 255, 255),
    }
    for camera_name, camera_data in camera_manifest.items():
        metadata_path = args.camera_manifest.parent / camera_data["metadata_path"]
        camera_metadata = json.load(open(metadata_path))
        camera_K = np.asarray(camera_metadata["K"], dtype=np.float64)
        camera_width, camera_height = camera_metadata["resolution"]
        X_WC_blender = np.asarray(camera_data["X_WC"], dtype=np.float64)
        X_WC = X_WC_blender @ X_OPENCV_CAMERA_FROM_BLENDER_CAMERA
        add_camera_frustum(
            ply,
            camera_K,
            camera_width,
            camera_height,
            X_WC,
            camera_colors.get(camera_name, (255, 255, 255)),
        )

    # ── Write ─────────────────────────────────────────────────────────────────
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ply.write(args.out)
    print(f"Scene exported to {args.out}")


if __name__ == "__main__":
    main()
