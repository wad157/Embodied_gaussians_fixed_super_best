#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import trimesh
import xacro

REPO = Path(__file__).resolve().parents[1]
DVRK_ROOT = REPO / "data/super/dvrk_model"
SOURCE_XACRO = DVRK_ROOT / "urdf/Classic/PSM1.urdf.xacro"
OUT_DIR = REPO / "data/super/psm_robot"
NATIVE_DIR = REPO / "data/super/grasp5_native"
OLD_URDF = OUT_DIR / "psm.urdf"

INPUT_JOINT_NAMES = [
    "outer_yaw",
    "outer_pitch",
    "outer_insertion",
    "outer_roll",
    "outer_wrist_pitch",
    "outer_wrist_yaw",
    "jaw",
]
INPUT_TO_URDF_JOINT = {
    "outer_yaw": "yaw",
    "outer_pitch": "pitch",
    "outer_insertion": "insertion",
    "outer_roll": "roll",
    "outer_wrist_pitch": "wrist_pitch",
    "outer_wrist_yaw": "wrist_yaw",
    "jaw": "jaw",
}
DEFAULT_BASE_ORIGIN = {
    "rpy": "-0.238342 -0.478597 2.466444",
    "xyz": "0.132451 -0.155286 0.139112",
}


def parse_current_base_origin() -> dict[str, str]:
    if not OLD_URDF.exists():
        return DEFAULT_BASE_ORIGIN.copy()
    root = ET.parse(OLD_URDF).getroot()
    fixed = root.find("./joint[@name='fixed']/origin")
    if fixed is None:
        return DEFAULT_BASE_ORIGIN.copy()
    return {
        "rpy": fixed.attrib.get("rpy", DEFAULT_BASE_ORIGIN["rpy"]),
        "xyz": fixed.attrib.get("xyz", DEFAULT_BASE_ORIGIN["xyz"]),
    }


def prepare_resolved_xacro_tree(tmp_root: Path) -> None:
    src_urdf = DVRK_ROOT / "urdf"
    dst_urdf = tmp_root / "urdf"
    for src in src_urdf.rglob("*.xacro"):
        rel = src.relative_to(src_urdf)
        dst = dst_urdf / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_text()
        text = text.replace("$(find dvrk_model)", str(tmp_root))
        dst.write_text(text)


def expand_xacro(tmp_root: Path) -> ET.Element:
    expanded = xacro.process_file(str(tmp_root / "urdf/Classic/PSM1.urdf.xacro"))
    return ET.fromstring(expanded.toxml())


def mesh_from_scene(obj) -> trimesh.Trimesh:
    if isinstance(obj, trimesh.Trimesh):
        return obj
    if isinstance(obj, trimesh.Scene):
        meshes = []
        for geom in obj.geometry.values():
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0:
                meshes.append(geom)
        if not meshes:
            raise ValueError("scene contains no mesh geometry")
        return trimesh.util.concatenate(meshes)
    raise TypeError(f"unsupported mesh object {type(obj)!r}")


def source_mesh_path(filename: str, tmp_root: Path) -> Path:
    if filename.startswith("package://dvrk_model/"):
        rel = filename[len("package://dvrk_model/"):]
        return DVRK_ROOT / rel
    path = Path(filename)
    if path.is_absolute():
        try:
            rel = path.relative_to(tmp_root)
            return DVRK_ROOT / rel
        except ValueError:
            return path
    return (DVRK_ROOT / filename).resolve()


def convert_mesh(src: Path, out_mesh_dir: Path) -> str:
    out_mesh_dir.mkdir(parents=True, exist_ok=True)
    out = out_mesh_dir / f"{src.stem.lower()}.stl"
    if out.exists():
        return f"meshes/{out.name}"
    if src.suffix.lower() == ".stl":
        shutil.copy2(src, out)
    else:
        mesh = mesh_from_scene(trimesh.load(src, process=False))
        mesh.export(out)
    return f"meshes/{out.name}"


def clone_element(elem: ET.Element) -> ET.Element:
    return ET.fromstring(ET.tostring(elem, encoding="unicode"))


def ensure_collision_and_inertial(root: ET.Element) -> None:
    for link in root.findall("link"):
        name = link.attrib["name"]
        if name == "world":
            continue
        if link.find("collision") is None:
            visuals = link.findall("visual")
            if visuals:
                for visual in visuals:
                    collision = ET.Element("collision")
                    origin = visual.find("origin")
                    geom = visual.find("geometry")
                    if origin is not None:
                        collision.append(clone_element(origin))
                    if geom is not None:
                        collision.append(clone_element(geom))
                    link.append(collision)
            else:
                collision = ET.Element("collision")
                ET.SubElement(collision, "origin", {"rpy": "0 0 0", "xyz": "0 0 0"})
                geom = ET.SubElement(collision, "geometry")
                if name.endswith("tool_tip_link"):
                    ET.SubElement(geom, "sphere", {"radius": "0.001"})
                else:
                    ET.SubElement(geom, "box", {"size": "0.001 0.001 0.001"})
                link.append(collision)
        if link.find("inertial") is None:
            inertial = ET.Element("inertial")
            ET.SubElement(inertial, "mass", {"value": "0.001"})
            ET.SubElement(
                inertial,
                "inertia",
                {
                    "ixx": "1e-8",
                    "ixy": "0",
                    "ixz": "0",
                    "iyy": "1e-8",
                    "iyz": "0",
                    "izz": "1e-8",
                },
            )
            link.append(inertial)


def widen_limits(root: ET.Element) -> list[str]:
    changed = []
    updates = {
        "roll": ("-3.5", "3.5"),
        "jaw": ("-1.2", "1.6"),
    }
    for joint_name, (lower, upper) in updates.items():
        joint = root.find(f"./joint[@name='{joint_name}']")
        if joint is None:
            continue
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
        limit.set("lower", lower)
        limit.set("upper", upper)
        if limit.get("velocity") is None:
            limit.set("velocity", ".4")
        if limit.get("effort") is None:
            limit.set("effort", "1000")
        changed.append(f"{joint_name}: [{lower}, {upper}]")
    return changed


def rewrite_meshes(root: ET.Element, tmp_root: Path, out_mesh_dir: Path) -> dict[str, str]:
    converted = {}
    for mesh_elem in root.findall(".//mesh"):
        original = mesh_elem.attrib["filename"]
        src = source_mesh_path(original, tmp_root)
        if not src.exists():
            raise FileNotFoundError(f"mesh source not found: {original} -> {src}")
        rel = convert_mesh(src, out_mesh_dir)
        mesh_elem.set("filename", rel)
        converted[str(src)] = rel
    return converted


def extract_and_remove_mimic(root: ET.Element) -> dict[str, dict[str, float | str]]:
    mimic_map = {}
    for joint in root.findall("joint"):
        mimic = joint.find("mimic")
        if mimic is None:
            continue
        mimic_map[joint.attrib["name"]] = {
            "source": mimic.attrib["joint"],
            "multiplier": float(mimic.attrib.get("multiplier", "1")),
            "offset": float(mimic.attrib.get("offset", "0")),
        }
        joint.remove(mimic)
    return mimic_map


def patch_base_origin(root: ET.Element, origin: dict[str, str]) -> None:
    fixed = root.find("./joint[@name='fixed']")
    if fixed is None:
        raise ValueError("expanded URDF has no fixed world->base joint")
    origin_elem = fixed.find("origin")
    if origin_elem is None:
        origin_elem = ET.SubElement(fixed, "origin")
    origin_elem.set("rpy", origin["rpy"])
    origin_elem.set("xyz", origin["xyz"])


def reachable_links(root: ET.Element) -> set[str]:
    children: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        children.setdefault(parent, []).append(child)
    seen = {"world"}
    stack = ["world"]
    while stack:
        parent = stack.pop()
        for child in children.get(parent, []):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def clean_old_outputs() -> list[str]:
    removed = []
    if OUT_DIR.exists():
        for child in OUT_DIR.iterdir():
            removed.append(str(child.relative_to(REPO)))
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for png in sorted(NATIVE_DIR.glob("psm*.png")):
        removed.append(str(png.relative_to(REPO)))
        png.unlink()
    return removed


def main() -> None:
    base_origin = parse_current_base_origin()
    removed = clean_old_outputs()
    tmp_root = Path(tempfile.mkdtemp(prefix="super_dvrk_xacro_"))
    try:
        prepare_resolved_xacro_tree(tmp_root)
        root = expand_xacro(tmp_root)
        mimic_map = extract_and_remove_mimic(root)
        patch_base_origin(root, base_origin)
        limit_changes = widen_limits(root)
        converted = rewrite_meshes(root, tmp_root, OUT_DIR / "meshes")
        ensure_collision_and_inertial(root)
        root.insert(0, ET.Comment("Generated by scripts/rebuild_super_psm_from_dvrk_xacro.py from dVRK Classic PSM1 xacro."))
        root.insert(1, ET.Comment("Mimic tags are exported to psm_mimic_map.json and removed from URDF for parser compatibility."))
        ET.indent(root, space="  ")
        ET.ElementTree(root).write(OUT_DIR / "psm.urdf", encoding="utf-8", xml_declaration=True)

        link_names = [link.attrib["name"] for link in root.findall("link")]
        joint_names = [joint.attrib["name"] for joint in root.findall("joint")]
        seen = reachable_links(root)
        visual_links = [link.attrib["name"] for link in root.findall("link") if link.find("visual") is not None]
        collision_links = [link.attrib["name"] for link in root.findall("link") if link.find("collision") is not None]
        unreachable = sorted(set(link_names) - seen)

        driven_joint_names = []
        for name in [INPUT_TO_URDF_JOINT[n] for n in INPUT_JOINT_NAMES]:
            driven_joint_names.append(name)
        for name in joint_names:
            if name in mimic_map:
                driven_joint_names.append(name)

        (OUT_DIR / "psm_mimic_map.json").write_text(json.dumps({
            "source": "dVRK Classic PSM1 xacro mimic tags",
            "input_joint_names": INPUT_JOINT_NAMES,
            "input_to_urdf_joint": INPUT_TO_URDF_JOINT,
            "mimic": mimic_map,
            "driven_joint_names_recommended_order": driven_joint_names,
        }, indent=2))

        report = {
            "source_xacro": str(SOURCE_XACRO.relative_to(REPO)),
            "base_origin_from_previous_right_handed_urdf": base_origin,
            "removed_old_outputs": removed,
            "link_count": len(link_names),
            "joint_count": len(joint_names),
            "joint_names": joint_names,
            "visual_link_count": len(visual_links),
            "collision_link_count": len(collision_links),
            "mimic_joint_count": len(mimic_map),
            "mimic_joint_names": sorted(mimic_map),
            "unreachable_links_from_world": unreachable,
            "converted_mesh_count": len(converted),
            "limit_changes": limit_changes,
            "urdf": str((OUT_DIR / "psm.urdf").relative_to(REPO)),
            "mimic_map": str((OUT_DIR / "psm_mimic_map.json").relative_to(REPO)),
        }
        (OUT_DIR / "psm_rebuild_report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
