#!/usr/bin/env python3
"""CPU gates for dense annotation-free AllTracker tissue queries."""

from __future__ import annotations

import json

import numpy as np

from track_super_alltracker_surface_particles import (
    backproject_to_table,
    tissue_grid_queries,
)


def main() -> None:
    tissue = np.full((30, 42), 255, dtype=np.uint8)
    tool = np.zeros_like(tissue)
    tool[10:20, 10:20] = 255
    depth = np.full(tissue.shape, 0.1, dtype=np.float32)
    intrinsic = np.asarray(
        [[100.0, 0.0, 20.0], [0.0, 100.0, 15.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    x_table_camera = np.eye(4, dtype=np.float64)
    grid_pixels = np.asarray(
        [[x, y] for y in range(3, 30, 6) for x in range(3, 42, 6)],
        dtype=np.float64,
    )
    surface = backproject_to_table(
        grid_pixels,
        np.full(len(grid_pixels), 0.1),
        intrinsic,
        x_table_camera,
    )
    particle_ids = np.arange(100, 100 + len(surface), dtype=np.int32)
    pixels, selected_ids, lifted = tissue_grid_queries(
        tissue_mask=tissue,
        tool_mask=tool,
        depth=depth,
        intrinsic=intrinsic,
        x_table_camera=x_table_camera,
        surface_positions_table=surface,
        surface_particle_ids=particle_ids,
        spacing_px=6,
        maximum_surface_distance_m=1.0e-6,
        maximum_queries_per_particle=2,
    )
    selected_tool = np.asarray(
        [tool[int(round(v)), int(round(u))] > 0 for u, v in pixels]
    )
    nearest = np.linalg.norm(
        lifted[:, None] - surface[None], axis=2
    ).min(axis=1)
    gates = {
        "grid_is_dense": bool(len(pixels) >= 30),
        "tool_region_is_excluded": bool(not selected_tool.any()),
        "every_query_gets_an_immutable_surface_id": bool(
            len(selected_ids) == len(pixels)
            and np.isin(selected_ids, particle_ids).all()
        ),
        "depth_lift_matches_surface_gate": bool(
            np.all(nearest <= 1.0e-6)
        ),
        "per_particle_query_cap_is_respected": bool(
            max(np.count_nonzero(selected_ids == value) for value in selected_ids)
            <= 2
        ),
    }
    report = {"gates": gates, "passed": all(gates.values())}
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
