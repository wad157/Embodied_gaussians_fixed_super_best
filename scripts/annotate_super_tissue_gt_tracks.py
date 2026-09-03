#!/usr/bin/env python3
"""Human-verify ten SUPER tissue landmarks for independent evaluation.

The first scheduled frame is annotated from scratch.  Later scheduled frames
receive forward/backward LK proposals propagated through every intervening
video frame.  A proposal is never ground truth by itself: the annotator must
explicitly press ENTER after checking the complete frame.  Visible, occluded,
and out-of-view states are stored separately following TAP-Vid-style practice.

This file deliberately does not import the simulator, online residual mapper,
or the existing automatically generated calibration tracklets.  The resulting
annotations are held-out evaluation data, not a training signal.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RGB_DIR = REPO_ROOT / "data/super/grasp5_native/rgb"
DEFAULT_METADATA = (
    REPO_ROOT / "data/super/grasp5_offline_demo/videos/stereo_left.json"
)
DEFAULT_CALIBRATION = REPO_ROOT / "data/super/grasp5_native/calib_rectified.json"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "data/super/tissue_calibration_v1/stage_a_frozen_manifest.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data/super/evaluation_v1/manual_tissue_tracks_10/annotations.json"
)
SCHEMA = "super_tissue_manual_track_ground_truth_v1"
VALID_STATUSES = {"visible", "occluded", "out_of_view"}
POINT_COUNT = 10
POINT_COLORS = (
    (48, 48, 255),
    (30, 160, 255),
    (30, 230, 230),
    (60, 220, 60),
    (255, 190, 40),
    (255, 90, 40),
    (220, 50, 190),
    (180, 80, 255),
    (255, 210, 180),
    (230, 230, 230),
)
REGION_HINTS = (
    "near_contact",
    "near_contact",
    "near_contact",
    "near_contact",
    "mid_field",
    "mid_field",
    "mid_field",
    "far_field",
    "far_field",
    "far_field",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgb-dir", type=Path, default=DEFAULT_RGB_DIR)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=1439)
    parser.add_argument("--frame-stride", type=int, default=10)
    parser.add_argument("--point-count", type=int, default=POINT_COUNT)
    parser.add_argument(
        "--edit-point-id",
        type=int,
        choices=range(POINT_COUNT),
        metavar="0..9",
        help=(
            "Re-annotate exactly one zero-based evaluation point from the first "
            "scheduled frame while locking all other observations. For example, "
            "--edit-point-id 8 edits evaluation point_id=8 (display label T09)."
        ),
    )
    parser.add_argument("--tracking-scale", type=float, default=0.5)
    parser.add_argument("--maximum-fb-error-px", type=float, default=2.0)
    parser.add_argument("--window-width", type=int, default=1800)
    parser.add_argument("--window-height", type=int, default=1000)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing annotation without opening a window.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check inputs and print the exact annotation schedule.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def required_frames_from_manifest(manifest: dict[str, Any]) -> list[int]:
    landmarks = manifest["sequence"]["landmarks"]
    return sorted(
        {
            int(value["left_frame"])
            for value in landmarks.values()
            if "left_frame" in value
        }
    )


def build_frame_schedule(
    *,
    start: int,
    end: int,
    stride: int,
    required_frames: list[int],
    excluded_frames: list[int],
) -> list[int]:
    if start < 0 or end < start:
        raise ValueError("Invalid inclusive frame range")
    if stride <= 0:
        raise ValueError("frame-stride must be positive")
    excluded = set(int(value) for value in excluded_frames)
    frames = set(range(start, end + 1, stride))
    frames.add(end)
    frames.update(
        frame for frame in required_frames if start <= frame <= end
    )

    # Include both sides of the exact chronological 80/20 boundary.  For an
    # inclusive range with N frames, future prediction starts at start+floor(.8N).
    frame_count = end - start + 1
    future_start = start + int(np.floor(0.8 * frame_count))
    frames.update((future_start - 1, future_start))
    return sorted(
        frame
        for frame in frames
        if start <= frame <= end and frame not in excluded
    )


def image_path(rgb_dir: Path, frame: int) -> Path:
    return rgb_dir / f"{frame:06d}-left.png"


def read_image(rgb_dir: Path, frame: int) -> np.ndarray:
    path = image_path(rgb_dir, frame)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return image


def read_gray_scaled(
    rgb_dir: Path, frame: int, scale: float
) -> np.ndarray:
    image = cv2.imread(
        str(image_path(rgb_dir, frame)), cv2.IMREAD_GRAYSCALE
    )
    if image is None:
        raise FileNotFoundError(image_path(rgb_dir, frame))
    if scale != 1.0:
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    return image


def point_definition(point_id: int) -> dict[str, Any]:
    return {
        "point_id": point_id,
        "name": f"T{point_id + 1:02d}",
        "object": "tissue",
        "region_hint": REGION_HINTS[point_id],
    }


def empty_observation(point_id: int) -> dict[str, Any]:
    return {
        "point_id": point_id,
        "status": "unresolved",
        "uv": None,
        "source": "unresolved",
        "fb_error_px": None,
    }


def new_payload(
    *,
    args: argparse.Namespace,
    metadata: dict[str, Any],
    manifest: dict[str, Any],
    schedule: list[int],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    future_start = args.start_frame + int(
        np.floor(0.8 * (args.end_frame - args.start_frame + 1))
    )
    return {
        "schema": SCHEMA,
        "status": "in_progress",
        "created_utc": utc_now(),
        "updated_utc": utc_now(),
        "point_count": args.point_count,
        "coordinate_frame": (
            "full-resolution rectified stereo-left pixels; origin top-left; "
            "u points right and v points down"
        ),
        "image_size_wh": list(image_size),
        "selection_protocol": {
            "T01-T04": "near-contact textured tissue landmarks",
            "T05-T07": "mid-field textured tissue landmarks",
            "T08-T10": "far-field textured tissue landmarks",
            "avoid": (
                "tool pixels, highlights, smooth textureless patches, tissue "
                "boundaries, and points already used by automatic calibration"
            ),
        },
        "sampling": {
            "frame_range_inclusive": [args.start_frame, args.end_frame],
            "frame_stride": args.frame_stride,
            "scheduled_frames": schedule,
            "scheduled_frame_count": len(schedule),
            "future_prediction_split": {
                "ratio": [0.8, 0.2],
                "train_frame_range_inclusive": [args.start_frame, future_start - 1],
                "test_frame_range_inclusive": [future_start, args.end_frame],
            },
            "stereo_excluded_frames": manifest["sequence"]["offline_stereo"]
            ["stereo_excluded_left_frames"],
        },
        "proposal_method": {
            "name": "forward_backward_pyramidal_LK",
            "tracking_scale": args.tracking_scale,
            "maximum_fb_error_px_full_resolution": args.maximum_fb_error_px,
            "warning": (
                "Proposals are not ground truth until the frame is explicitly "
                "human-verified. This tracker is not used by the evaluated model."
            ),
        },
        "source": {
            "rgb_directory": str(args.rgb_dir.resolve()),
            "left_metadata": {
                "path": str(args.metadata.resolve()),
                "sha256": sha256(args.metadata),
            },
            "stereo_calibration": {
                "path": str(args.calibration.resolve()),
                "sha256": sha256(args.calibration),
            },
            "stage_a_manifest": {
                "path": str(args.manifest.resolve()),
                "sha256": sha256(args.manifest),
            },
            "timestamps_s": [
                float(metadata["timestamps"][frame]) for frame in schedule
            ],
        },
        "points": [point_definition(index) for index in range(args.point_count)],
        "frames": {},
        "usage": (
            "Held-out 2D/3D tracking evaluation only. Never feed these labels "
            "to visual residuals, stiffness fitting, parameter selection, or "
            "automatic tracker training."
        ),
    }


def validate_payload(
    payload: dict[str, Any], *, require_complete: bool = False
) -> dict[str, Any]:
    errors: list[str] = []
    if payload.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    point_count = int(payload.get("point_count", -1))
    if point_count != POINT_COUNT:
        errors.append(f"point_count must be exactly {POINT_COUNT}")
    points = payload.get("points", [])
    if [row.get("point_id") for row in points] != list(range(point_count)):
        errors.append("point definitions are not contiguous 0..9")
    schedule = payload.get("sampling", {}).get("scheduled_frames", [])
    if schedule != sorted(set(schedule)):
        errors.append("scheduled frames must be sorted and unique")
    width, height = payload.get("image_size_wh", [-1, -1])
    verified_count = 0
    visible_count = 0
    status_counts = {status: 0 for status in VALID_STATUSES}
    for frame in schedule:
        record = payload.get("frames", {}).get(f"{frame:06d}")
        if record is None:
            continue
        observations = record.get("observations", [])
        if len(observations) != point_count:
            errors.append(f"frame {frame}: expected {point_count} observations")
            continue
        ids = [int(item.get("point_id", -1)) for item in observations]
        if ids != list(range(point_count)):
            errors.append(f"frame {frame}: point IDs are not ordered 0..9")
        valid_frame = True
        for item in observations:
            status = item.get("status")
            if status not in VALID_STATUSES:
                if record.get("human_verified", False):
                    errors.append(f"frame {frame}: unresolved verified point")
                valid_frame = False
                continue
            status_counts[status] += 1
            uv = item.get("uv")
            if status in {"visible", "occluded"}:
                if not isinstance(uv, list) or len(uv) != 2:
                    errors.append(f"frame {frame}: {status} point lacks uv")
                    valid_frame = False
                elif not (
                    np.isfinite(uv).all()
                    and 0 <= float(uv[0]) < width
                    and 0 <= float(uv[1]) < height
                ):
                    errors.append(f"frame {frame}: point outside image")
                    valid_frame = False
            elif uv is not None:
                errors.append(f"frame {frame}: out_of_view point must have uv=null")
                valid_frame = False
            if status == "visible":
                visible_count += 1
        if record.get("human_verified", False) and valid_frame:
            verified_count += 1
    complete = verified_count == len(schedule) and not errors
    if require_complete and not complete:
        errors.append(
            f"annotation incomplete: {verified_count}/{len(schedule)} frames verified"
        )
    return {
        "passed": not errors,
        "complete": complete,
        "scheduled_frame_count": len(schedule),
        "verified_frame_count": verified_count,
        "visible_observation_count": visible_count,
        "status_counts": status_counts,
        "errors": errors,
    }


def save_payload(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_utc"] = utc_now()
    validation = validate_payload(payload)
    payload["status"] = "complete" if validation["complete"] else "in_progress"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def propagate_points_lk(
    *,
    rgb_dir: Path,
    source_frame: int,
    target_frame: int,
    source_observations: list[dict[str, Any]],
    scale: float,
    maximum_fb_error_full_px: float,
) -> list[dict[str, Any]]:
    if target_frame <= source_frame:
        raise ValueError("LK proposals require a later target frame")
    proposals = [empty_observation(index) for index in range(len(source_observations))]
    active_ids: list[int] = []
    active_points: list[list[float]] = []
    for item in source_observations:
        if item.get("status") == "visible" and item.get("uv") is not None:
            active_ids.append(int(item["point_id"]))
            active_points.append(
                (np.asarray(item["uv"], dtype=np.float32) * scale).tolist()
            )
        elif item.get("status") == "occluded" and item.get("uv") is not None:
            point_id = int(item["point_id"])
            proposals[point_id] = {
                "point_id": point_id,
                "status": "unresolved",
                "uv": list(item["uv"]),
                "source": "previous_occlusion_requires_review",
                "fb_error_px": None,
            }

    if not active_ids:
        return proposals
    points = np.asarray(active_points, dtype=np.float32)
    alive = np.ones(len(points), dtype=bool)
    worst_fb = np.zeros(len(points), dtype=np.float32)
    previous = read_gray_scaled(rgb_dir, source_frame, scale)
    threshold_scaled = maximum_fb_error_full_px * scale
    for frame in range(source_frame + 1, target_frame + 1):
        current = read_gray_scaled(rgb_dir, frame, scale)
        live_indices = np.flatnonzero(alive)
        if len(live_indices):
            source = points[live_indices].reshape(-1, 1, 2)
            forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
                previous,
                current,
                source,
                None,
                winSize=(31, 31),
                maxLevel=4,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    30,
                    0.01,
                ),
            )
            if forward is None:
                alive[live_indices] = False
            else:
                backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                    current,
                    previous,
                    forward,
                    None,
                    winSize=(31, 31),
                    maxLevel=4,
                    criteria=(
                        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                        30,
                        0.01,
                    ),
                )
                if backward is None:
                    alive[live_indices] = False
                else:
                    tracked = forward.reshape(-1, 2)
                    error = np.linalg.norm(
                        backward.reshape(-1, 2) - source.reshape(-1, 2), axis=1
                    )
                    height, width = current.shape
                    good = (
                        (forward_status.reshape(-1) > 0)
                        & (backward_status.reshape(-1) > 0)
                        & np.isfinite(tracked).all(axis=1)
                        & np.isfinite(error)
                        & (error <= threshold_scaled)
                        & (tracked[:, 0] >= 0)
                        & (tracked[:, 0] < width)
                        & (tracked[:, 1] >= 0)
                        & (tracked[:, 1] < height)
                    )
                    points[live_indices[good]] = tracked[good]
                    worst_fb[live_indices[good]] = np.maximum(
                        worst_fb[live_indices[good]], error[good] / scale
                    )
                    alive[live_indices[~good]] = False
        previous = current

    for local_index, point_id in enumerate(active_ids):
        uv = (points[local_index] / scale).astype(float).tolist()
        proposals[point_id] = {
            "point_id": point_id,
            "status": "visible" if alive[local_index] else "unresolved",
            "uv": uv,
            "source": (
                "lk_proposal_pending_human_verification"
                if alive[local_index]
                else "lk_failed_requires_manual_review"
            ),
            "fb_error_px": (
                float(worst_fb[local_index]) if alive[local_index] else None
            ),
        }
    return proposals


class TissueTrackAnnotator:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        metadata: dict[str, Any],
        payload: dict[str, Any],
    ) -> None:
        self.args = args
        self.metadata = metadata
        self.payload = payload
        self.schedule = [int(value) for value in payload["sampling"]["scheduled_frames"]]
        self.edit_point_id = args.edit_point_id
        self.window_name = "SUPER 10-point tissue ground-truth annotation"
        if self.edit_point_id is not None:
            self.window_name += (
                f" | ONLY point_id={self.edit_point_id} "
                f"(T{self.edit_point_id + 1:02d})"
            )
        self.header_height = 118
        self.history: list[dict[str, Any]] = []
        self.selected_id: int | None = None
        self.session_verified_indices: set[int] = set()
        self.locked_observations = self.capture_locked_observations()
        self.current_index = (
            0 if self.edit_point_id is not None else self.first_incomplete_index()
        )
        self.image = np.empty((0, 0, 3), dtype=np.uint8)
        self.current: dict[str, Any] = {}
        self.scale = 1.0
        self.display_width = 0
        self.display_height = 0
        self.load_current()

    def capture_locked_observations(self) -> dict[str, dict[int, dict[str, Any]]]:
        if self.edit_point_id is None:
            return {}
        locked: dict[str, dict[int, dict[str, Any]]] = {}
        for frame in self.schedule:
            key = f"{frame:06d}"
            record = self.payload.get("frames", {}).get(key)
            if record is None:
                raise RuntimeError(
                    "Single-point editing requires an existing complete annotation; "
                    f"frame {frame} is missing"
                )
            locked[key] = {
                int(item["point_id"]): copy.deepcopy(item)
                for item in record.get("observations", [])
                if int(item["point_id"]) != self.edit_point_id
            }
            if len(locked[key]) != self.args.point_count - 1:
                raise RuntimeError(
                    f"frame {frame}: cannot lock the other "
                    f"{self.args.point_count - 1} observations"
                )
        return locked

    def assert_locked_observations_unchanged(self) -> None:
        if self.edit_point_id is None:
            return
        key = f"{self.frame:06d}"
        expected = self.locked_observations[key]
        actual = {
            int(item["point_id"]): item
            for item in self.current["observations"]
            if int(item["point_id"]) != self.edit_point_id
        }
        if actual != expected:
            raise RuntimeError(
                f"Safety check failed: a locked observation changed at frame {self.frame}"
            )

    def first_incomplete_index(self) -> int:
        frames = self.payload.get("frames", {})
        for index, frame in enumerate(self.schedule):
            record = frames.get(f"{frame:06d}")
            if record is None or not record.get("human_verified", False):
                return index
        return len(self.schedule) - 1

    @property
    def frame(self) -> int:
        return self.schedule[self.current_index]

    def previous_verified(self) -> tuple[int, dict[str, Any]] | None:
        for index in range(self.current_index - 1, -1, -1):
            frame = self.schedule[index]
            record = self.payload["frames"].get(f"{frame:06d}")
            if record is not None and record.get("human_verified", False):
                return frame, record
        return None

    def fresh_record(self) -> dict[str, Any]:
        previous = self.previous_verified()
        if previous is None:
            observations: list[dict[str, Any]] = []
        else:
            source_frame, source = previous
            observations = propagate_points_lk(
                rgb_dir=self.args.rgb_dir,
                source_frame=source_frame,
                target_frame=self.frame,
                source_observations=source["observations"],
                scale=self.args.tracking_scale,
                maximum_fb_error_full_px=self.args.maximum_fb_error_px,
            )
        return {
            "frame": self.frame,
            "timestamp_s": float(self.metadata["timestamps"][self.frame]),
            "human_verified": False,
            "verified_utc": None,
            "observations": observations,
        }

    def load_current(self, *, reset_proposal: bool = False) -> None:
        self.image = read_image(self.args.rgb_dir, self.frame)
        height, width = self.image.shape[:2]
        self.scale = min(
            self.args.window_width / width,
            (self.args.window_height - self.header_height) / height,
            1.0,
        )
        self.display_width = int(round(width * self.scale))
        self.display_height = int(round(height * self.scale))
        key = f"{self.frame:06d}"
        existing = self.payload["frames"].get(key)
        if self.edit_point_id is not None:
            if existing is None:
                raise RuntimeError(
                    "Single-point editing cannot create missing frame records"
                )
            self.current = copy.deepcopy(existing)
            needs_proposal = self.current_index > 0 and (
                reset_proposal
                or self.current_index not in self.session_verified_indices
            )
            if needs_proposal:
                previous = self.previous_verified()
                if previous is None:
                    raise RuntimeError("No preceding verified frame for LK proposal")
                source_frame, source = previous
                proposals = propagate_points_lk(
                    rgb_dir=self.args.rgb_dir,
                    source_frame=source_frame,
                    target_frame=self.frame,
                    source_observations=source["observations"],
                    scale=self.args.tracking_scale,
                    maximum_fb_error_full_px=self.args.maximum_fb_error_px,
                )
                target = copy.deepcopy(proposals[self.edit_point_id])
                self.current["observations"][self.edit_point_id] = target
                self.current["human_verified"] = False
                self.current["verified_utc"] = None
        elif existing is None or reset_proposal:
            self.current = self.fresh_record()
        else:
            self.current = copy.deepcopy(existing)
        self.history.clear()
        self.selected_id = self.edit_point_id

    def snapshot(self) -> None:
        self.history.append(copy.deepcopy(self.current))
        if len(self.history) > 50:
            self.history.pop(0)

    def undo(self) -> None:
        if self.history:
            self.current = self.history.pop()

    def set_unverified(self) -> None:
        self.current["human_verified"] = False
        self.current["verified_utc"] = None

    def observation(self, point_id: int) -> dict[str, Any] | None:
        for item in self.current["observations"]:
            if int(item["point_id"]) == point_id:
                return item
        return None

    def set_point_visible(self, point_id: int, uv: list[float]) -> None:
        if self.edit_point_id is not None and point_id != self.edit_point_id:
            return
        self.snapshot()
        item = self.observation(point_id)
        value = {
            "point_id": point_id,
            "status": "visible",
            "uv": [float(uv[0]), float(uv[1])],
            "source": "manual",
            "fb_error_px": None,
        }
        if item is None:
            self.current["observations"].append(value)
            self.current["observations"].sort(key=lambda row: row["point_id"])
        else:
            item.update(value)
        self.selected_id = point_id
        self.set_unverified()

    def set_status(self, status: str) -> None:
        if self.selected_id is None or status not in VALID_STATUSES:
            return
        if (
            self.edit_point_id is not None
            and self.selected_id != self.edit_point_id
        ):
            return
        item = self.observation(self.selected_id)
        if item is None:
            return
        if status == "visible" and item.get("uv") is None:
            return
        self.snapshot()
        item["status"] = status
        item["source"] = "manual_status"
        item["fb_error_px"] = None
        if status == "out_of_view":
            item["uv"] = None
        self.set_unverified()

    def map_click(self, x: int, y: int) -> list[float] | None:
        y_image = y - self.header_height
        if not (
            0 <= x < self.display_width
            and 0 <= y_image < self.display_height
        ):
            return None
        return [x / self.scale, y_image / self.scale]

    def closest_point(self, uv: list[float]) -> int | None:
        candidates = []
        for item in self.current["observations"]:
            if item.get("uv") is None:
                continue
            distance = float(
                np.linalg.norm(np.asarray(item["uv"]) - np.asarray(uv))
            )
            candidates.append((distance, int(item["point_id"])))
        if not candidates:
            return None
        distance, point_id = min(candidates)
        return point_id if distance * self.scale <= 24.0 else None

    def on_mouse(
        self, event: int, x: int, y: int, _flags: int, _parameter: object
    ) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        uv = self.map_click(x, y)
        if uv is None:
            return
        if self.edit_point_id is not None:
            self.set_point_visible(self.edit_point_id, uv)
            return
        if len(self.current["observations"]) < self.args.point_count:
            point_id = len(self.current["observations"])
            self.set_point_visible(point_id, uv)
            return
        if self.selected_id is None:
            closest = self.closest_point(uv)
            if closest is not None:
                self.selected_id = closest
                return
        if self.selected_id is not None:
            self.set_point_visible(self.selected_id, uv)

    def current_errors(self) -> list[str]:
        observations = self.current["observations"]
        errors = []
        if len(observations) != self.args.point_count:
            errors.append(f"need {self.args.point_count - len(observations)} more points")
            return errors
        height, width = self.image.shape[:2]
        for point_id in range(self.args.point_count):
            item = self.observation(point_id)
            if item is None or item.get("status") not in VALID_STATUSES:
                errors.append(f"T{point_id + 1:02d} unresolved")
                continue
            if item["status"] in {"visible", "occluded"}:
                uv = item.get("uv")
                if uv is None or not (
                    0 <= float(uv[0]) < width and 0 <= float(uv[1]) < height
                ):
                    errors.append(f"T{point_id + 1:02d} needs an in-image position")
        return errors

    def persist_current(self, *, verified: bool) -> bool:
        errors = self.current_errors()
        if verified and errors:
            return False
        self.assert_locked_observations_unchanged()
        self.current["human_verified"] = bool(verified)
        self.current["verified_utc"] = utc_now() if verified else None
        if verified:
            for item in self.current["observations"]:
                if item["source"] == "lk_proposal_pending_human_verification":
                    item["source"] = "lk_proposal_human_accepted"
        self.payload["frames"][f"{self.frame:06d}"] = copy.deepcopy(self.current)
        save_payload(self.args.output, self.payload)
        if verified and self.edit_point_id is not None:
            self.session_verified_indices.add(self.current_index)
        return True

    def advance(self) -> None:
        if not self.persist_current(verified=True):
            return
        if self.current_index < len(self.schedule) - 1:
            self.current_index += 1
            self.load_current()

    def retreat(self) -> None:
        if self.current_index > 0:
            self.persist_current(verified=bool(self.current["human_verified"]))
            self.current_index -= 1
            self.load_current()

    def select_from_key(self, key: int) -> bool:
        if self.edit_point_id is not None:
            if ord("0") <= key <= ord("9"):
                self.selected_id = self.edit_point_id
                return True
            return False
        if ord("1") <= key <= ord("9"):
            self.selected_id = key - ord("1")
            return True
        if key == ord("0"):
            self.selected_id = 9
            return True
        return False

    def draw_point(self, canvas: np.ndarray, item: dict[str, Any]) -> None:
        uv = item.get("uv")
        point_id = int(item["point_id"])
        if uv is None:
            return
        xy = (
            int(round(float(uv[0]) * self.scale)),
            self.header_height + int(round(float(uv[1]) * self.scale)),
        )
        color = POINT_COLORS[point_id]
        selected = point_id == self.selected_id
        radius = 9 if selected else 7
        status = item.get("status")
        if status == "visible":
            cv2.circle(canvas, xy, radius + 3, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, xy, radius, color, -1, cv2.LINE_AA)
        elif status == "occluded":
            cv2.drawMarker(
                canvas, xy, color, cv2.MARKER_TILTED_CROSS, 22, 3, cv2.LINE_AA
            )
        else:
            cv2.drawMarker(
                canvas, xy, (0, 165, 255), cv2.MARKER_CROSS, 22, 3, cv2.LINE_AA
            )
        label = f"T{point_id + 1:02d}"
        cv2.putText(
            canvas,
            label,
            (xy[0] + 9, xy[1] - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (xy[0] + 9, xy[1] - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )

    def draw_magnifier(self, canvas: np.ndarray) -> None:
        if self.selected_id is None:
            return
        item = self.observation(self.selected_id)
        if item is None or item.get("uv") is None:
            return
        u, v = (int(round(value)) for value in item["uv"])
        half = 45
        padded = cv2.copyMakeBorder(
            self.image, half, half, half, half, cv2.BORDER_REFLECT_101
        )
        crop = padded[v : v + 2 * half + 1, u : u + 2 * half + 1]
        size = min(260, self.display_height // 3)
        zoom = cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)
        center = size // 2
        cv2.drawMarker(
            zoom,
            (center, center),
            POINT_COLORS[self.selected_id],
            cv2.MARKER_CROSS,
            28,
            2,
            cv2.LINE_AA,
        )
        x0 = self.display_width - size - 10
        y0 = self.header_height + self.display_height - size - 10
        canvas[y0 : y0 + size, x0 : x0 + size] = zoom
        cv2.rectangle(canvas, (x0, y0), (x0 + size, y0 + size), (255, 255, 255), 2)

    def render(self) -> np.ndarray:
        resized = cv2.resize(
            self.image,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA,
        )
        canvas = np.zeros(
            (self.header_height + self.display_height, self.display_width, 3),
            dtype=np.uint8,
        )
        canvas[self.header_height :] = resized
        validation = validate_payload(self.payload)
        status = "VERIFIED" if self.current.get("human_verified") else "NOT VERIFIED"
        selected = "none" if self.selected_id is None else f"T{self.selected_id + 1:02d}"
        line1 = (
            f"SUPER tissue GT | frame {self.frame} | {self.current_index + 1}/"
            f"{len(self.schedule)} | {status} | selected={selected} | "
            f"done={validation['verified_frame_count']}/{len(self.schedule)}"
        )
        if self.edit_point_id is not None:
            line2 = (
                f"SINGLE-POINT MODE: only point_id={self.edit_point_id} "
                f"(T{self.edit_point_id + 1:02d}) is editable; other 9 points are locked. "
                "Click its correct position, then press ENTER."
            )
        elif len(self.current["observations"]) < self.args.point_count:
            next_id = len(self.current["observations"])
            line2 = (
                f"FIRST FRAME: click T{next_id + 1:02d} ({REGION_HINTS[next_id]}). "
                "T01-T04 near contact, T05-T07 middle, T08-T10 far."
            )
        else:
            unresolved = [
                f"T{item['point_id'] + 1:02d}"
                for item in self.current["observations"]
                if item.get("status") not in VALID_STATUSES
            ]
            line2 = (
                "Check every marker. Click marker to select; click elsewhere to move it. "
                + (f"UNRESOLVED: {','.join(unresolved)}" if unresolved else "All statuses resolved.")
            )
        if self.edit_point_id is not None:
            line3 = (
                "Keys: left-click move | O occluded | X out-of-view | V visible | "
                "ENTER verify+next | P previous | R reset LK proposal | U undo | Q save+quit"
            )
        else:
            line3 = (
                "Keys: 1..9/0 select | left-click move/restore | O occluded | X out-of-view "
                "| V visible | ENTER verify+next | P previous | R reset proposal | U undo | Q save+quit"
            )
        for text, y, color in (
            (line1, 28, (245, 245, 245)),
            (line2, 65, (60, 220, 255)),
            (line3, 101, (210, 210, 210)),
        ):
            cv2.putText(
                canvas,
                text,
                (12, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                color,
                1,
                cv2.LINE_AA,
            )
        for item in self.current["observations"]:
            self.draw_point(canvas, item)
        out_of_view = [
            f"T{item['point_id'] + 1:02d}"
            for item in self.current["observations"]
            if item.get("status") == "out_of_view"
        ]
        if out_of_view:
            cv2.putText(
                canvas,
                "OUT OF VIEW: " + ",".join(out_of_view),
                (12, self.header_height + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
        self.draw_magnifier(canvas)
        return canvas

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.window_name,
            self.display_width,
            self.header_height + self.display_height,
        )
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        try:
            while True:
                cv2.imshow(self.window_name, self.render())
                key = cv2.waitKeyEx(30)
                if key < 0:
                    continue
                key &= 0xFF
                if self.select_from_key(key):
                    continue
                if key in (13, 10):
                    self.advance()
                elif key in (ord("p"), ord("P")):
                    self.retreat()
                elif key in (ord("u"), ord("U")):
                    self.undo()
                elif key in (ord("o"), ord("O")):
                    self.set_status("occluded")
                elif key in (ord("x"), ord("X")):
                    self.set_status("out_of_view")
                elif key in (ord("v"), ord("V")):
                    self.set_status("visible")
                elif key in (ord("r"), ord("R")):
                    if not self.current.get("human_verified", False):
                        self.load_current(reset_proposal=True)
                elif key in (ord("q"), ord("Q"), 27):
                    self.persist_current(
                        verified=bool(self.current.get("human_verified", False))
                    )
                    break
        finally:
            cv2.destroyAllWindows()


def check_inputs(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any], list[int], tuple[int, int]]:
    for path in (args.rgb_dir, args.metadata, args.calibration, args.manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.point_count != POINT_COUNT:
        raise ValueError(
            f"This evaluation protocol requires exactly {POINT_COUNT} points"
        )
    if not 0.0 < args.tracking_scale <= 1.0:
        raise ValueError("tracking-scale must be in (0,1]")
    metadata = load_json(args.metadata)
    manifest = load_json(args.manifest)
    resolution = tuple(int(value) for value in metadata["resolution"])
    if args.end_frame >= len(metadata["timestamps"]):
        raise ValueError("end-frame exceeds stereo-left metadata")
    schedule = build_frame_schedule(
        start=args.start_frame,
        end=args.end_frame,
        stride=args.frame_stride,
        required_frames=required_frames_from_manifest(manifest),
        excluded_frames=manifest["sequence"]["offline_stereo"]
        ["stereo_excluded_left_frames"],
    )
    for frame in schedule:
        if not image_path(args.rgb_dir, frame).is_file():
            raise FileNotFoundError(image_path(args.rgb_dir, frame))
    first = read_image(args.rgb_dir, schedule[0])
    actual_resolution = (first.shape[1], first.shape[0])
    if actual_resolution != resolution:
        raise RuntimeError(
            f"Image resolution {actual_resolution} != metadata {resolution}"
        )
    return metadata, manifest, schedule, resolution


def verify_source_identity(payload: dict[str, Any], args: argparse.Namespace) -> None:
    expected = {
        "left_metadata": (args.metadata, sha256(args.metadata)),
        "stereo_calibration": (args.calibration, sha256(args.calibration)),
        "stage_a_manifest": (args.manifest, sha256(args.manifest)),
    }
    for name, (_path, digest) in expected.items():
        recorded = payload.get("source", {}).get(name, {}).get("sha256")
        if recorded != digest:
            raise RuntimeError(f"Source identity mismatch for {name}")


def main() -> None:
    args = parse_args()
    metadata, manifest, schedule, resolution = check_inputs(args)
    if args.dry_run:
        future = args.start_frame + int(
            np.floor(0.8 * (args.end_frame - args.start_frame + 1))
        )
        print(
            json.dumps(
                {
                    "passed": True,
                    "point_count": args.point_count,
                    "image_size_wh": resolution,
                    "frame_range_inclusive": [args.start_frame, args.end_frame],
                    "frame_stride": args.frame_stride,
                    "scheduled_frame_count": len(schedule),
                    "scheduled_frames": schedule,
                    "future_prediction_train_end": future - 1,
                    "future_prediction_test_start": future,
                    "edit_point_id": args.edit_point_id,
                    "output": str(args.output.resolve()),
                },
                indent=2,
            )
        )
        return

    if args.output.exists():
        payload = load_json(args.output)
        verify_source_identity(payload, args)
        if payload.get("sampling", {}).get("scheduled_frames") != schedule:
            raise RuntimeError(
                "Existing annotation schedule differs from the requested one; "
                "use a different --output path instead of mixing protocols"
            )
        if args.edit_point_id is not None:
            report = validate_payload(payload, require_complete=True)
            if not report["passed"]:
                raise RuntimeError(
                    "Single-point editing requires a complete valid base annotation: "
                    + "; ".join(report["errors"])
                )
    else:
        payload = new_payload(
            args=args,
            metadata=metadata,
            manifest=manifest,
            schedule=schedule,
            image_size=resolution,
        )
        save_payload(args.output, payload)

    if args.validate_only:
        report = validate_payload(payload, require_complete=True)
        print(json.dumps(report, indent=2))
        if not report["passed"]:
            raise SystemExit(1)
        return

    TissueTrackAnnotator(args=args, metadata=metadata, payload=payload).run()
    final = validate_payload(payload)
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
