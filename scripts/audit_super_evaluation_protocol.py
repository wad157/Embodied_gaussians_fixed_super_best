#!/usr/bin/env python3
"""Fail closed unless the frozen SUPER evaluation matches the paper protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "data/super/evaluation_v1/manual_tissue_tracks_10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--report", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    annotations_path = root / "annotations.json"
    ground_truth_path = root / "ground_truth_2d3d_v1.npz"
    manifest_path = root / "ground_truth_2d3d_v1.json"
    depth_root = root / "stereo_depth_v1"
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(ground_truth_path, allow_pickle=False) as archive:
        gt = {name: np.asarray(archive[name]) for name in archive.files}

    require(annotations["status"] == "complete", "Annotations are incomplete")
    require(bool(manifest["passed"]), "Frozen 2D/3D manifest did not pass")
    require(
        sha256(annotations_path) == manifest["annotations"]["sha256"],
        "Annotation hash differs from the frozen 3D lift",
    )
    scheduled = np.asarray(
        annotations["sampling"]["scheduled_frames"], dtype=np.int32
    )
    require(
        np.array_equal(scheduled, gt["frame_indices"]),
        "Manual and lifted frame schedules differ",
    )
    require(
        len(scheduled) == int(annotations["sampling"]["scheduled_frame_count"]),
        "Manual schedule count is inconsistent",
    )
    point_count = int(annotations["point_count"])
    require(point_count == 10, "Formal SUPER protocol requires ten landmarks")
    require(gt["uv"].shape == (len(scheduled), point_count, 2), "GT UV shape is invalid")

    allowed_sources = {"manual", "lk_proposal_human_accepted"}
    source_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for slot, frame_index in enumerate(scheduled.tolist()):
        record = annotations["frames"].get(f"{frame_index:06d}")
        require(record is not None, f"Missing manual frame {frame_index}")
        require(bool(record.get("human_verified")), f"Frame {frame_index} is not human verified")
        observations = sorted(record["observations"], key=lambda item: int(item["point_id"]))
        require(
            [int(item["point_id"]) for item in observations]
            == list(range(point_count)),
            f"Frame {frame_index} does not contain every landmark",
        )
        for point_id, observation in enumerate(observations):
            source = str(observation["source"])
            status = str(observation["status"])
            require(source in allowed_sources, f"Unverified point source: {source}")
            require(status in {"visible", "occluded", "out_of_view"}, f"Invalid point status: {status}")
            source_counts[source] = source_counts.get(source, 0) + 1
            status_counts[status] = status_counts.get(status, 0) + 1
            if status == "visible":
                require(
                    np.allclose(
                        np.asarray(observation["uv"], dtype=np.float32),
                        gt["uv"][slot, point_id],
                        atol=1.0e-5,
                        rtol=0.0,
                    ),
                    f"Lifted UV differs from manual UV at {frame_index}/{point_id}",
                )

    require(bool(gt["valid_3d"].all()), "Formal GT has missing strict stereo 3D")
    require(
        bool((gt["depth_sampling_radius_px"] == 0).all()),
        "A formal 3D point did not use its exact annotated pixel",
    )
    intrinsic = gt["K_left_rect"].astype(np.float64)
    x_world_camera = gt["X_world_camera_opencv"].astype(np.float64)
    maximum_camera_error_m = 0.0
    maximum_world_error_m = 0.0
    for slot, frame_index in enumerate(scheduled.tolist()):
        depth = np.load(
            depth_root / f"{frame_index:06d}-depth_high_confidence.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        for point_id in range(point_count):
            u, v = gt["uv"][slot, point_id].astype(np.float64)
            z = float(depth[int(round(v)), int(round(u))])
            require(np.isfinite(z) and z > 0.0, f"Missing strict depth at {frame_index}/{point_id}")
            camera = np.asarray(
                [
                    (u - intrinsic[0, 2]) * z / intrinsic[0, 0],
                    (v - intrinsic[1, 2]) * z / intrinsic[1, 1],
                    z,
                ],
                dtype=np.float64,
            )
            world = (x_world_camera @ np.r_[camera, 1.0])[:3]
            maximum_camera_error_m = max(
                maximum_camera_error_m,
                float(np.linalg.norm(camera - gt["xyz_camera_m"][slot, point_id])),
            )
            maximum_world_error_m = max(
                maximum_world_error_m,
                float(np.linalg.norm(world - gt["xyz_world_m"][slot, point_id])),
            )
    require(maximum_camera_error_m <= 1.0e-6, "Camera backprojection is inconsistent")
    require(maximum_world_error_m <= 1.0e-6, "World backprojection is inconsistent")

    sequence_start = int(scheduled.min())
    sequence_end = int(scheduled.max())
    full_frames = np.arange(sequence_start, sequence_end + 1, dtype=np.int32)
    reconstruction_test = full_frames[full_frames % 8 == 0]
    reconstruction_train = full_frames[full_frames % 8 != 0]
    future_start = int(gt["future_test_start_frame"].item())
    future_train = full_frames[full_frames < future_start]
    future_test = full_frames[full_frames >= future_start]
    require(
        (len(reconstruction_train), len(reconstruction_test)) == (1260, 180),
        "Reconstruction split is not exact 7:1 over 1440 frames",
    )
    require(
        (len(future_train), len(future_test)) == (1152, 288),
        "Future split is not exact chronological 80:20",
    )
    require(
        set(reconstruction_train).isdisjoint(reconstruction_test),
        "Reconstruction train/test leakage",
    )
    require(set(future_train).isdisjoint(future_test), "Future train/test leakage")

    report = {
        "schema": "super_evaluation_protocol_audit_v1",
        "passed": True,
        "manual_2d": {
            "frame_count": int(len(scheduled)),
            "point_count": point_count,
            "human_verified_observations": int(sum(source_counts.values())),
            "source_counts": source_counts,
            "status_counts": status_counts,
        },
        "stereo_3d": {
            "valid_count": int(gt["valid_3d"].sum()),
            "exact_pixel_count": int((gt["depth_sampling_radius_px"] == 0).sum()),
            "maximum_camera_backprojection_error_m": maximum_camera_error_m,
            "maximum_world_backprojection_error_m": maximum_world_error_m,
            "policy": manifest["depth"]["policy"],
        },
        "splits": {
            "reconstruction_7to1": {
                "train_frames": len(reconstruction_train),
                "test_frames": len(reconstruction_test),
                "test_rule": "frame_index % 8 == 0",
            },
            "future_80to20": {
                "train_frames": len(future_train),
                "test_frames": len(future_test),
                "test_start": future_start,
            },
        },
        "ground_truth_sha256": sha256(ground_truth_path),
    }
    report_path = (
        args.report.resolve()
        if args.report is not None
        else root / "protocol_audit_v1.json"
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
