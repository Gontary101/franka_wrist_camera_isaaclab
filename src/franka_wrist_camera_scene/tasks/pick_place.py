"""Pick-and-place task definitions."""

from __future__ import annotations

from dataclasses import dataclass

from .base import TaskSpec


@dataclass(frozen=True, slots=True)
class PickPlaceTaskSpec(TaskSpec):
    """Static single-object pick-and-place task."""

    object_name: str = "target_cube"
    ee_body_name: str = "panda_hand"
    instruction: str = "pick up the object and place it on the target area"

    object_pos_local: tuple[float, float, float] = (0.58, -0.16, 1.08)
    place_pos_local: tuple[float, float, float] = (0.55, 0.22, 1.08)
    grasp_closing_axis_xy: tuple[float, float] | None = None

    object_local_bbox_min: tuple[float, float, float] | None = None
    object_local_bbox_max: tuple[float, float, float] | None = None

    object_transit_clearance_m: float = 0.13
    pregrasp_clearance_m: float = 0.055
    top_grasp_depth_m: float = 0.025
    place_transit_clearance_m: float = 0.13
    support_surface_z_local: float = 1.05
    object_bottom_clearance_m: float = 0.006
    place_pregrasp_clearance_m: float = 0.055

    lift_height_m: float = 0.20
    open_finger_m: float = 0.04
    closed_finger_m: float = 0.0

    free_space_max_speed_m_s: float = 0.22
    free_space_max_accel_m_s2: float = 0.45

    approach_max_speed_m_s: float = 0.08
    approach_max_accel_m_s2: float = 0.20

    lift_max_speed_m_s: float = 0.12
    lift_max_accel_m_s2: float = 0.25

    retreat_max_speed_m_s: float = 0.15
    retreat_max_accel_m_s2: float = 0.30

    grasp_dwell_s: float = 1.0
    release_dwell_s: float = 1.0


def instruction_for_object(object_label: str) -> str:
    return f"pick up the {object_label} and place it on the target area"


def make_pick_place_episode_spec(
    base_spec: PickPlaceTaskSpec,
    object_xy_offset: tuple[float, float],
    place_xy_offset: tuple[float, float],
    object_label: str,
    grasp_closing_axis_xy: tuple[float, float] | None = None,
    object_local_bbox_min: tuple[float, float, float] | None = None,
    object_local_bbox_max: tuple[float, float, float] | None = None,
) -> PickPlaceTaskSpec:
    resolved_bbox_min = (
        object_local_bbox_min
        if object_local_bbox_min is not None
        else base_spec.object_local_bbox_min
    )
    if resolved_bbox_min is not None:
        object_root_z = (
            base_spec.support_surface_z_local
            - resolved_bbox_min[2]
            + base_spec.object_bottom_clearance_m
        )
    else:
        object_root_z = base_spec.object_pos_local[2]

    object_pos = (
        base_spec.object_pos_local[0] + object_xy_offset[0],
        base_spec.object_pos_local[1] + object_xy_offset[1],
        object_root_z,
    )
    place_pos = (
        base_spec.place_pos_local[0] + place_xy_offset[0],
        base_spec.place_pos_local[1] + place_xy_offset[1],
        object_root_z,
    )

    return PickPlaceTaskSpec(
        instruction=instruction_for_object(object_label),
        object_name=base_spec.object_name,
        ee_body_name=base_spec.ee_body_name,
        object_pos_local=object_pos,
        place_pos_local=place_pos,
        grasp_closing_axis_xy=(
            grasp_closing_axis_xy
            if grasp_closing_axis_xy is not None
            else base_spec.grasp_closing_axis_xy
        ),
        object_local_bbox_min=(
            object_local_bbox_min
            if object_local_bbox_min is not None
            else base_spec.object_local_bbox_min
        ),
        object_local_bbox_max=(
            object_local_bbox_max
            if object_local_bbox_max is not None
            else base_spec.object_local_bbox_max
        ),
        object_transit_clearance_m=base_spec.object_transit_clearance_m,
        pregrasp_clearance_m=base_spec.pregrasp_clearance_m,
        top_grasp_depth_m=base_spec.top_grasp_depth_m,
        place_transit_clearance_m=base_spec.place_transit_clearance_m,
        support_surface_z_local=base_spec.support_surface_z_local,
        object_bottom_clearance_m=base_spec.object_bottom_clearance_m,
        place_pregrasp_clearance_m=base_spec.place_pregrasp_clearance_m,
        lift_height_m=base_spec.lift_height_m,
        open_finger_m=base_spec.open_finger_m,
        closed_finger_m=base_spec.closed_finger_m,
        free_space_max_speed_m_s=base_spec.free_space_max_speed_m_s,
        free_space_max_accel_m_s2=base_spec.free_space_max_accel_m_s2,
        approach_max_speed_m_s=base_spec.approach_max_speed_m_s,
        approach_max_accel_m_s2=base_spec.approach_max_accel_m_s2,
        lift_max_speed_m_s=base_spec.lift_max_speed_m_s,
        lift_max_accel_m_s2=base_spec.lift_max_accel_m_s2,
        retreat_max_speed_m_s=base_spec.retreat_max_speed_m_s,
        retreat_max_accel_m_s2=base_spec.retreat_max_accel_m_s2,
        grasp_dwell_s=base_spec.grasp_dwell_s,
        release_dwell_s=base_spec.release_dwell_s,
    )
