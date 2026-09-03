#!/usr/bin/env python3
"""CPU unit gates for fixed flow-depth physical-particle range bindings."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from embodied_gaussians.physics_simulator.flow_depth_particle_observer import (  # noqa: E402
    FlowDepthObservationSettings,
    FlowDepthParticleRangeBindings,
    FlowDepthRangeBindingSettings,
    FlowDepthTriangleBindingSettings,
    FlowDepthStateUpdateSettings,
    compute_flow_depth_particle_state_update,
    backproject_pixels,
    build_fixed_particle_range_bindings,
    build_fixed_particle_triangle_bindings,
    observe_flow_depth_tracks,
    sample_depth_near_pixels,
)


def main() -> None:
    depth = np.full((9, 9), np.nan, dtype=np.float32)
    depth[4, 4] = 1.0
    sampled, radii = sample_depth_near_pixels(
        depth,
        np.asarray([[4.0, 4.0], [5.0, 4.0], [-1.0, 0.0]]),
        maximum_radius_px=1,
    )
    intrinsic = np.eye(3, dtype=np.float64)
    points = backproject_pixels(
        np.asarray([[0.0, 0.0], [1.0, 2.0]]),
        np.asarray([1.0, 2.0]),
        intrinsic,
        np.eye(4),
    )

    xy = np.asarray(
        [(x, y) for y in range(5) for x in range(5)], dtype=np.float64
    )
    rest = np.column_stack((xy * 0.001, np.ones(len(xy))))
    surface = np.ones(len(rest), dtype=bool)
    fixed = np.zeros(len(rest), dtype=bool)
    fixed[0] = True
    settings = FlowDepthRangeBindingSettings(
        radius_m=0.0005,
        fallback_radius_m=0.0011,
        sigma_m=0.001,
        minimum_movable_particles=3,
        maximum_depth_sampling_radius_px=1,
    )
    pixels = np.asarray([[2.0, 2.0], [4.0, 4.0], [0.0, 0.0]])
    synthetic_depth = np.ones((5, 5), dtype=np.float32)
    requested = np.asarray([True, True, False])
    bindings_a = build_fixed_particle_range_bindings(
        initial_pixels_uv=pixels,
        depth=synthetic_depth,
        intrinsic=np.asarray(
            [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        x_table_camera=np.eye(4),
        rest_positions_table=rest,
        surface_mask=surface,
        fixed_mask=fixed,
        initial_track_valid=requested,
        settings=settings,
    )
    bindings_b = build_fixed_particle_range_bindings(
        initial_pixels_uv=pixels,
        depth=synthetic_depth,
        intrinsic=np.asarray(
            [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        x_table_camera=np.eye(4),
        rest_positions_table=rest,
        surface_mask=surface,
        fixed_mask=fixed,
        initial_track_valid=requested,
        settings=settings,
    )
    center_track = 0
    count = int(bindings_a.support_counts[center_track])
    ids = bindings_a.particle_ids[center_track, :count]
    weights = bindings_a.particle_weights[center_track, :count]
    distances = np.linalg.norm(
        rest[ids] - bindings_a.initial_points_table[center_track],
        axis=1,
    )
    monotonic = np.all(
        weights[np.argsort(distances)][:-1]
        >= weights[np.argsort(distances)][1:] - 1.0e-7
    )

    triangle_rest = np.asarray(
        [[0.0, 0.0, 1.0], [0.002, 0.0, 1.0], [0.0, 0.002, 1.0]],
        dtype=np.float64,
    )
    triangle_depth = np.ones((3, 3), dtype=np.float32)
    triangle_depth[1, 0] = 1.0015
    triangle_depth[0, 1] = 1.0030
    triangle_bindings = build_fixed_particle_triangle_bindings(
        initial_pixels_uv=np.asarray(
            [[0.4, 0.4], [0.4, 1.2], [1.2, 0.4]], dtype=np.float64
        ),
        depth=triangle_depth,
        intrinsic=np.asarray(
            [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        x_table_camera=np.eye(4),
        rest_positions_table=triangle_rest,
        surface_faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        surface_mask=np.ones(3, dtype=bool),
        fixed_mask=np.zeros(3, dtype=bool),
        initial_track_valid=np.ones(3, dtype=bool),
        settings=FlowDepthTriangleBindingSettings(
            primary_maximum_surface_distance_m=0.001,
            fallback_maximum_surface_distance_m=0.002,
            minimum_movable_vertices=3,
            maximum_depth_sampling_radius_px=0,
        ),
    )
    triangle_valid = triangle_bindings.track_valid
    triangle_centres = np.einsum(
        "ni,nij->nj",
        triangle_bindings.particle_weights[triangle_valid],
        triangle_rest[triangle_bindings.particle_ids[triangle_valid]],
    )

    observed = observe_flow_depth_tracks(
        current_pixels_uv=np.asarray([[2.0, 2.0], [3.0, 3.0], [0.0, 0.0]]),
        next_pixels_uv=np.asarray([[3.0, 2.0], [4.0, 3.0], [1.0, 0.0]]),
        current_depth=synthetic_depth,
        next_depth=synthetic_depth,
        intrinsic=np.asarray(
            [[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        x_table_camera=np.eye(4),
        current_track_valid=np.asarray([True, True, True]),
        next_track_valid=np.asarray([True, True, False]),
        confidence=np.asarray([1.0, 0.5, 1.0]),
        settings=FlowDepthObservationSettings(
            maximum_depth_sampling_radius_px=0,
            minimum_depth_m=0.5,
            maximum_depth_m=1.5,
            maximum_depth_change_m=0.1,
            maximum_observed_flow_m=0.1,
        ),
    )

    simple_bindings = FlowDepthParticleRangeBindings(
        track_valid=np.asarray([True]),
        particle_ids=np.asarray([[0, 1]], dtype=np.int32),
        particle_weights=np.asarray([[0.5, 0.5]], dtype=np.float32),
        support_counts=np.asarray([2], dtype=np.int32),
        movable_support_counts=np.asarray([2], dtype=np.int32),
        binding_radius_m=np.asarray([0.006], dtype=np.float32),
        initial_depth_m=np.asarray([1.0], dtype=np.float32),
        depth_sampling_radius_px=np.asarray([0], dtype=np.int16),
        initial_points_table=np.asarray([[0.0005, 0.0, 0.0]], dtype=np.float32),
        nearest_surface_distance_m=np.asarray([0.0005], dtype=np.float32),
    )
    simple_bindings.validate()
    simple_observation = type(observed)(
        track_valid=np.asarray([True]),
        confidence=np.asarray([1.0], dtype=np.float32),
        current_depth_m=np.asarray([1.0], dtype=np.float32),
        next_depth_m=np.asarray([1.0], dtype=np.float32),
        current_depth_sampling_radius_px=np.asarray([0], dtype=np.int16),
        next_depth_sampling_radius_px=np.asarray([0], dtype=np.int16),
        current_points_table=np.asarray([[0.0005, 0.0, 0.0]], dtype=np.float32),
        next_points_table=np.asarray([[0.0015, 0.0, 0.0]], dtype=np.float32),
        observed_flow_table=np.asarray([[0.001, 0.0, 0.0]], dtype=np.float32),
    )
    current_q = np.asarray([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]])
    predicted_q = current_q + np.asarray([0.0002, 0.0, 0.0])
    update = compute_flow_depth_particle_state_update(
        bindings=simple_bindings,
        observation=simple_observation,
        current_positions=current_q,
        predicted_positions=predicted_q,
        predicted_velocities=np.zeros_like(current_q),
        particle_inverse_masses=np.ones(2),
        observation_dt_s=0.1,
        settings=FlowDepthStateUpdateSettings(
            position_gain=0.5,
            velocity_gain=0.1,
            maximum_position_correction_m=0.01,
            maximum_velocity_correction_m_s=1.0,
        ),
    )
    fixed_update = compute_flow_depth_particle_state_update(
        bindings=simple_bindings,
        observation=simple_observation,
        current_positions=current_q,
        predicted_positions=predicted_q,
        predicted_velocities=np.zeros_like(current_q),
        particle_inverse_masses=np.asarray([0.0, 1.0]),
        observation_dt_s=0.1,
        settings=FlowDepthStateUpdateSettings(
            position_gain=0.5,
            velocity_gain=0.1,
            maximum_position_correction_m=0.0003,
            maximum_velocity_correction_m_s=1.0,
        ),
    )
    gates = {
        "exact_depth_preferred": bool(
            np.isclose(sampled[0], 1.0) and radii[0] == 0
        ),
        "nearby_depth_fallback_works": bool(
            np.isclose(sampled[1], 1.0) and radii[1] == 1
        ),
        "out_of_bounds_depth_rejected": bool(
            not np.isfinite(sampled[2]) and radii[2] == -1
        ),
        "backprojection_identity_is_exact": bool(
            np.allclose(points, [[0.0, 0.0, 1.0], [2.0, 4.0, 2.0]])
        ),
        "requested_tracks_are_bound": bool(
            np.array_equal(bindings_a.track_valid, requested)
        ),
        "unrequested_track_has_no_support": bool(
            bindings_a.support_counts[2] == 0
        ),
        "fallback_radius_is_used_when_primary_is_sparse": bool(
            np.isclose(bindings_a.binding_radius_m[0], 0.0011)
        ),
        "minimum_movable_support_is_met": bool(
            np.all(bindings_a.movable_support_counts[requested] >= 3)
        ),
        "weights_sum_to_one": bool(
            np.allclose(bindings_a.valid_weight_sums(), 1.0, atol=2.0e-6)
        ),
        "closer_particles_have_no_smaller_weight": bool(monotonic),
        "binding_is_deterministic": bool(
            np.array_equal(bindings_a.particle_ids, bindings_b.particle_ids)
            and np.array_equal(
                bindings_a.particle_weights, bindings_b.particle_weights
            )
        ),
        "triangle_distance_gate_uses_face_not_vertex_distance": bool(
            np.array_equal(triangle_valid, [True, True, False])
            and np.isclose(
                triangle_bindings.binding_radius_m[0], 0.001
            )
            and np.isclose(
                triangle_bindings.binding_radius_m[1], 0.002
            )
        ),
        "triangle_binding_uses_exactly_three_face_vertices": bool(
            np.array_equal(
                triangle_bindings.particle_ids[triangle_valid],
                [[0, 1, 2], [0, 1, 2]],
            )
            and np.all(triangle_bindings.support_counts[triangle_valid] == 3)
        ),
        "triangle_barycentric_projection_reconstructs_surface_point": bool(
            triangle_bindings.binding_method
            == "surface_triangle_barycentric"
            and np.allclose(
                triangle_centres,
                triangle_bindings.surface_projection_points_table[
                    triangle_valid
                ],
                atol=1.0e-7,
            )
            and np.allclose(
                triangle_bindings.particle_weights[triangle_valid].sum(axis=1),
                1.0,
                atol=1.0e-7,
            )
        ),
        "flow_depth_observation_lifts_pixel_motion_to_3d": bool(
            np.array_equal(observed.track_valid, [True, True, False])
            and np.allclose(
                observed.observed_flow_table[:2],
                [[0.001, 0.0, 0.0], [0.001, 0.0, 0.0]],
                atol=1.0e-7,
            )
            and np.allclose(observed.confidence, [1.0, 0.5, 0.0])
        ),
        "innovation_subtracts_pbd_prediction": bool(
            np.allclose(update.predicted_track_flow, [[0.0002, 0.0, 0.0]])
            and np.allclose(update.innovation, [[0.0008, 0.0, 0.0]])
        ),
        "joint_solver_applies_regularized_half_innovation": bool(
            np.allclose(
                update.position_correction,
                [
                    [0.00038461538, 0.0, 0.0],
                    [0.00038461538, 0.0, 0.0],
                ],
                atol=1.0e-7,
            )
        ),
        "velocity_is_solved_from_observed_velocity_not_position_step": bool(
            np.allclose(
                update.velocity_correction,
                [
                    [0.00096153846, 0.0, 0.0],
                    [0.00096153846, 0.0, 0.0],
                ],
                atol=1.0e-7,
            )
        ),
        "fixed_particle_is_immutable_and_movable_step_is_capped": bool(
            np.allclose(fixed_update.position_correction[0], 0.0)
            and np.isclose(
                np.linalg.norm(fixed_update.position_correction[1]), 0.0003
            )
        ),
    }
    report = {
        "schema": "super_flow_depth_range_binding_cpu_gate_v1",
        "passed": bool(all(gates.values())),
        "gates": gates,
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("Flow-depth range-binding CPU gate failed")


if __name__ == "__main__":
    main()
