"""Geometry-aware deterministic tabletop clutter sampling."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any

from franka_wrist_camera_scene.scene.object_context import (
    CatalogObjectContext,
    load_catalog_object_context,
)
from franka_wrist_camera_scene.tasks.placement_geometry import object_root_pose_on_support


@dataclass(frozen=True, slots=True)
class XYRange:
    x: tuple[float, float]
    y: tuple[float, float]


@dataclass(frozen=True, slots=True)
class FootprintDisk:
    xy: tuple[float, float]
    radius_m: float


@dataclass(frozen=True, slots=True)
class ClutterObjectSpec:
    prim_name: str
    context: CatalogObjectContext
    pos_local: tuple[float, float, float]
    footprint_radius_m: float


def parse_xy_range(config: dict[str, Any]) -> XYRange:
    x_range = config["x"]
    y_range = config["y"]

    parsed = XYRange(
        x=(float(x_range[0]), float(x_range[1])),
        y=(float(y_range[0]), float(y_range[1])),
    )

    if parsed.x[1] <= parsed.x[0]:
        raise ValueError(f"Invalid clutter x range: {x_range}")
    if parsed.y[1] <= parsed.y[0]:
        raise ValueError(f"Invalid clutter y range: {y_range}")

    return parsed


def xy_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def planar_footprint_radius_m(
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    margin_m: float,
) -> float:
    if margin_m < 0.0:
        raise ValueError(f"Footprint margin must be non-negative, got {margin_m}")

    size_x = float(bbox_max[0]) - float(bbox_min[0])
    size_y = float(bbox_max[1]) - float(bbox_min[1])

    if size_x <= 0.0 or size_y <= 0.0:
        raise ValueError(f"Invalid planar bbox size: size_x={size_x}, size_y={size_y}")

    return 0.5 * math.hypot(size_x, size_y) + margin_m


def footprints_overlap(a: FootprintDisk, b: FootprintDisk) -> bool:
    return xy_distance(a.xy, b.xy) < (a.radius_m + b.radius_m)


def footprint_inside_xy_range(candidate: FootprintDisk, xy_range: XYRange) -> bool:
    x, y = candidate.xy
    r = candidate.radius_m
    return (
        x - r >= xy_range.x[0]
        and x + r <= xy_range.x[1]
        and y - r >= xy_range.y[0]
        and y + r <= xy_range.y[1]
    )


def _grid_search_non_overlapping_xy(
    xy_range: XYRange,
    candidate_radius_m: float,
    occupied: tuple[FootprintDisk, ...],
    grid_step_m: float,
) -> tuple[float, float] | None:
    if grid_step_m <= 0.0:
        raise ValueError(f"grid_step_m must be positive, got {grid_step_m}")

    x = xy_range.x[0] + candidate_radius_m
    x_limit = xy_range.x[1] - candidate_radius_m
    while x <= x_limit + 1e-9:
        y = xy_range.y[0] + candidate_radius_m
        y_limit = xy_range.y[1] - candidate_radius_m
        while y <= y_limit + 1e-9:
            candidate = FootprintDisk(xy=(x, y), radius_m=candidate_radius_m)
            if footprint_inside_xy_range(candidate, xy_range) and all(
                not footprints_overlap(candidate, existing) for existing in occupied
            ):
                return (x, y)
            y += grid_step_m
        x += grid_step_m

    return None


def sample_non_overlapping_xy(
    rng: random.Random,
    xy_range: XYRange,
    candidate_radius_m: float,
    occupied: tuple[FootprintDisk, ...],
    max_attempts: int,
    grid_step_m: float = 0.01,
) -> tuple[float, float]:
    if candidate_radius_m <= 0.0:
        raise ValueError(f"candidate_radius_m must be positive, got {candidate_radius_m}")
    if max_attempts <= 0:
        raise ValueError(f"max_attempts must be positive, got {max_attempts}")

    for _ in range(max_attempts):
        xy = (
            rng.uniform(xy_range.x[0], xy_range.x[1]),
            rng.uniform(xy_range.y[0], xy_range.y[1]),
        )
        candidate = FootprintDisk(xy=xy, radius_m=candidate_radius_m)

        if not footprint_inside_xy_range(candidate, xy_range):
            continue

        if all(not footprints_overlap(candidate, existing) for existing in occupied):
            return xy

    grid_xy = _grid_search_non_overlapping_xy(
        xy_range=xy_range,
        candidate_radius_m=candidate_radius_m,
        occupied=occupied,
        grid_step_m=grid_step_m,
    )
    if grid_xy is not None:
        return grid_xy

    occupied_summary = [
        {"xy": disk.xy, "radius_m": disk.radius_m}
        for disk in occupied
    ]
    raise RuntimeError(
        "Failed to sample a non-overlapping clutter position "
        f"after {max_attempts} random attempts and a deterministic grid search. "
        f"candidate_radius_m={candidate_radius_m}, "
        f"xy_range={xy_range}, occupied={occupied_summary}"
    )


def footprint_for_context(
    context: CatalogObjectContext,
    margin_m: float,
) -> float:
    return planar_footprint_radius_m(
        bbox_min=context.geometry.local_bbox_min,
        bbox_max=context.geometry.local_bbox_max,
        margin_m=margin_m,
    )


def sample_clutter_objects(
    clutter_cfg: dict[str, Any],
    seed: int,
    episode_id: int,
    support_surface_z_local: float,
    object_bottom_clearance_m: float,
    target_object_context: CatalogObjectContext,
    target_object_xy: tuple[float, float],
    placement_target_context: CatalogObjectContext,
    placement_target_xy: tuple[float, float],
) -> tuple[ClutterObjectSpec, ...]:
    count = int(clutter_cfg["count"])
    if count != 3:
        raise ValueError(
            "This commit supports exactly 3 clutter objects. "
            f"Got clutter.count={count}."
        )

    rng = random.Random(seed + 200_000 + episode_id)

    xy_range = parse_xy_range(clutter_cfg["xy_range"])
    max_attempts = int(clutter_cfg["max_sampling_attempts"])

    occupied: list[FootprintDisk] = [
        FootprintDisk(
            xy=target_object_xy,
            radius_m=footprint_for_context(
                target_object_context,
                margin_m=float(clutter_cfg["object_margin_m"]),
            ),
        ),
        FootprintDisk(
            xy=placement_target_xy,
            radius_m=footprint_for_context(
                placement_target_context,
                margin_m=float(clutter_cfg["placement_target_margin_m"]),
            ),
        ),
    ]

    clutter_margin_m = float(clutter_cfg["clutter_margin_m"])
    sampled_slots: list[tuple[int, CatalogObjectContext, float]] = []

    for clutter_index in range(count):
        context = load_catalog_object_context(
            catalog_config=clutter_cfg["catalog_config"],
            geometry_config=clutter_cfg["geometry_config"],
            category_id="sample",
            variant_id="sample",
            split=clutter_cfg["split"],
            role=clutter_cfg["role"],
            required_affordances=tuple(clutter_cfg["required_affordances"]),
            required_grasp_strategy=clutter_cfg["required_grasp_strategy"],
            rng=rng,
        )
        footprint_radius = footprint_for_context(context, margin_m=clutter_margin_m)
        sampled_slots.append((clutter_index, context, footprint_radius))

    placement_order = sorted(sampled_slots, key=lambda item: (item[2], item[0]))
    placed_xy: dict[int, tuple[float, float]] = {}

    for clutter_index, context, footprint_radius in placement_order:
        xy = sample_non_overlapping_xy(
            rng=rng,
            xy_range=xy_range,
            candidate_radius_m=footprint_radius,
            occupied=tuple(occupied),
            max_attempts=max_attempts,
        )
        placed_xy[clutter_index] = xy
        occupied.append(FootprintDisk(xy=xy, radius_m=footprint_radius))

    specs: list[ClutterObjectSpec] = []
    for clutter_index, context, footprint_radius in sampled_slots:
        xy = placed_xy[clutter_index]
        pos_local = object_root_pose_on_support(
            xy_pos=xy,
            support_surface_z=support_surface_z_local,
            object_bbox_min_z=context.geometry.local_bbox_min[2],
            bottom_clearance_m=object_bottom_clearance_m,
        )
        specs.append(
            ClutterObjectSpec(
                prim_name=f"ClutterObject{clutter_index}",
                context=context,
                pos_local=pos_local,
                footprint_radius_m=footprint_radius,
            )
        )

    return tuple(specs)
