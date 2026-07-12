# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import warp as wp


@wp.kernel
def update_gaussians_transforms_kernel(
    gaussian_pos_rel_body: wp.array(dtype=wp.vec3f),  # type: ignore
    gaussian_quats_rel_body: wp.array(dtype=wp.vec4f),  # type: ignore (w, x, y, z)
    gaussian_body_ids: wp.array(dtype=wp.int32),  # type: ignore
    body_transform: wp.array(dtype=wp.transformf),  # type: ignore (xyz, qw qx qy qz),
    gaussian_pos: wp.array(dtype=wp.vec3f),  # type: ignore
    gaussian_quats: wp.array(dtype=wp.vec4f),  # type: ignore (w, x, y, z)
):
    tid = wp.tid()
    body_id = gaussian_body_ids[tid]
    if body_id == -1:
        return
    T_WB = body_transform[body_id]
    p_BG = gaussian_pos_rel_body[tid]
    r_BG = gaussian_quats_rel_body[tid]
    q_BG = wp.quatf(r_BG[1], r_BG[2], r_BG[3], r_BG[0])
    T_BG = wp.transformf(p_BG, q_BG)
    T_WG = wp.transform_multiply(T_WB, T_BG)
    gaussian_pos[tid] = wp.transform_get_translation(T_WG)
    q_WG = wp.transform_get_rotation(T_WG)
    gaussian_quats[tid] = wp.vec4(q_WG[3], q_WG[0], q_WG[1], q_WG[2])  # type: ignore


@wp.kernel
def update_visual_forces_kernel(
    kp: float,
    old_means: wp.array(dtype=wp.vec3f),  # type: ignore
    old_quats: wp.array(dtype=wp.quatf),  # type: ignore
    old_opacities: wp.array(dtype=wp.float32),  # type: ignore
    new_means: wp.array(dtype=wp.vec3f),  # type: ignore
    new_quats: wp.array(dtype=wp.quatf),  # type: ignore
    body_ids: wp.array(dtype=wp.int32),  # type: ignore
    body_q: wp.array(dtype=wp.transformf),  # type: ignore
    forces: wp.array(dtype=wp.vec3f),  # type: ignore
    moments: wp.array(dtype=wp.vec3f),  # type: ignore
):
    tid = wp.tid()
    body_id = body_ids[tid]
    if body_id == -1:
        return
    opacity = old_opacities[tid]
    displacement = new_means[tid] - old_means[tid]
    q_WO = old_quats[tid]
    q_WN = new_quats[tid]
    q_NW = wp.quat_inverse(q_WN)
    q_NO = wp.mul(q_NW, q_WO)
    axis = wp.vec3()
    angle = wp.float32(0.0)  # type: ignore
    wp.quat_to_axis_angle(q_NO, axis, angle)
    T_WB = body_q[body_id]
    com = wp.transform_get_translation(T_WB)
    force = opacity * kp * displacement
    r = old_means[tid] - com
    moment_from_force = wp.cross(r, force)
    moment = moment_from_force
    forces[tid] = force
    moments[tid] = moment


@wp.kernel
def apply_forces_kernel(
    dt: float,
    total_force: wp.array(dtype=wp.vec3f),  # type: ignore
    total_moment: wp.array(dtype=wp.vec3f),  # type: ignore
    body_ids: wp.array(dtype=wp.int32),  # type: ignore
    gaussian_counts: wp.array(dtype=wp.int32),  # type: ignore
    apply_physics_forces: wp.array(dtype=wp.int32),  # type: ignore
    normalize_by_gaussian_count: int,
    max_force: float,
    max_moment: float,
    body_f: wp.array(dtype=wp.spatial_vectorf),  # type: ignore
):
    tid = wp.tid()
    bid = body_ids[tid]
    if apply_physics_forces[tid] != 0:
        force = total_force[tid]
        moment = total_moment[tid]
        if normalize_by_gaussian_count != 0 and gaussian_counts[tid] > 0:
            divisor = float(gaussian_counts[tid])
            force = force / divisor
            moment = moment / divisor
        force_norm = wp.length(force)
        if max_force > 0.0 and force_norm > max_force:
            force = force * (max_force / force_norm)
        moment_norm = wp.length(moment)
        if max_moment > 0.0 and moment_norm > max_moment:
            moment = moment * (max_moment / moment_norm)
        body_f[bid] = wp.spatial_vector(moment, force)  # type: ignore
