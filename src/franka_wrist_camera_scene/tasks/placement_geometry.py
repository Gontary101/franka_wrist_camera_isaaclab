"""Geometry helpers for object placement on support surfaces."""

from __future__ import annotations


def object_root_z_on_support(
    support_surface_z: float,
    object_bbox_min_z: float,
    bottom_clearance_m: float,
) -> float:
    return support_surface_z - object_bbox_min_z + bottom_clearance_m


def object_root_pose_on_support(
    xy_pos: tuple[float, float],
    support_surface_z: float,
    object_bbox_min_z: float,
    bottom_clearance_m: float,
) -> tuple[float, float, float]:
    return (
        xy_pos[0],
        xy_pos[1],
        object_root_z_on_support(
            support_surface_z=support_surface_z,
            object_bbox_min_z=object_bbox_min_z,
            bottom_clearance_m=bottom_clearance_m,
        ),
    )
