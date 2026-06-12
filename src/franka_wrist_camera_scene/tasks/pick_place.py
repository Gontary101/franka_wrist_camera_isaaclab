"""Pick-and-place task definitions."""

from __future__ import annotations

from dataclasses import dataclass

from .base import TaskSpec
from .placement_geometry import object_root_pose_on_support, object_root_z_on_support


@dataclass(frozen=True, slots=True)
class PickPlaceTaskSpec(TaskSpec):
    """Static single-object pick-and-place task."""

    object_name: str = "target_cube"
    ee_body_name: str = "panda_hand"
    instruction: str = "pick up the object and place it on the target area"

    object_pos_local: tuple[float, float, float] = (0.58, -0.16, 1.08)
    place_pos_local: tuple[float, float, float] = (0.55, 0.22, 1.08)
    tcp_offset_local: tuple[float, float, float] = (0.0, 0.0, 0.10)
    grasp_closing_axis_xy: tuple[float, float] | None = None

    object_local_bbox_min: tuple[float, float, float] | None = None
    object_local_bbox_max: tuple[float, float, float] | None = None
    placement_target_pos_local: tuple[float, float, float] | None = None
    placement_target_local_bbox_min: tuple[float, float, float] | None = None
    placement_target_local_bbox_max: tuple[float, float, float] | None = None

    object_transit_clearance_m: float = 0.13
    pregrasp_clearance_m: float = 0.055
    top_grasp_depth_m: float = 0.025
    support_surface_z_local: float = 1.05
    object_bottom_clearance_m: float = 0.006
    place_pregrasp_clearance_m: float = 0.055
    receptacle_release_bottom_clearance_m: float = 0.015

    lift_height_m: float = 0.20
    open_finger_m: float = 0.04
    closed_finger_m: float = 0.0

    free_space_max_speed_m_s: float = 0.22
    approach_max_speed_m_s: float = 0.08
    lift_max_speed_m_s: float = 0.12
    retreat_max_speed_m_s: float = 0.15
    grasp_dwell_s: float = 1.0
    release_dwell_s: float = 1.0


def instruction_for_object(object_label: str) -> str:
    return f"pick up the {object_label} and place it on the target area"


def instruction_for_object_and_receptacle(object_label: str, placement_label: str) -> str:
    return f"pick up the {object_label} and place it in the {placement_label}"


def make_pick_place_episode_spec(
    base_spec: PickPlaceTaskSpec,
    object_xy_offset: tuple[float, float],
    place_xy_offset: tuple[float, float],
    object_label: str,
    grasp_closing_axis_xy: tuple[float, float] | None = None,
    object_local_bbox_min: tuple[float, float, float] | None = None,
    object_local_bbox_max: tuple[float, float, float] | None = None,
    placement_target_pos_local: tuple[float, float, float] | None = None,
    placement_target_local_bbox_min: tuple[float, float, float] | None = None,
    placement_target_local_bbox_max: tuple[float, float, float] | None = None,
    placement_label: str | None = None,
) -> PickPlaceTaskSpec:
    resolved_bbox_min = (
        object_local_bbox_min
        if object_local_bbox_min is not None
        else base_spec.object_local_bbox_min
    )
    resolved_bbox_max = (
        object_local_bbox_max
        if object_local_bbox_max is not None
        else base_spec.object_local_bbox_max
    )
    if resolved_bbox_min is None or resolved_bbox_max is None:
        raise ValueError("Pick-place episode specs require object bbox metadata.")

    resolved_placement_bbox_min = (
        placement_target_local_bbox_min
        if placement_target_local_bbox_min is not None
        else base_spec.placement_target_local_bbox_min
    )
    resolved_placement_bbox_max = (
        placement_target_local_bbox_max
        if placement_target_local_bbox_max is not None
        else base_spec.placement_target_local_bbox_max
    )
    resolved_placement_pos = (
        placement_target_pos_local
        if placement_target_pos_local is not None
        else base_spec.placement_target_pos_local
    )
    uses_receptacle = (
        resolved_placement_pos is not None
        or resolved_placement_bbox_min is not None
        or resolved_placement_bbox_max is not None
        or placement_label is not None
    )
    if uses_receptacle and (
        resolved_placement_pos is None
        or resolved_placement_bbox_min is None
        or resolved_placement_bbox_max is None
        or placement_label is None
    ):
        raise ValueError(
            "Receptacle pick-place episode specs require placement target pose, bbox metadata, and label."
        )

    object_xy = (
        base_spec.object_pos_local[0] + object_xy_offset[0],
        base_spec.object_pos_local[1] + object_xy_offset[1],
    )
    place_xy = (
        base_spec.place_pos_local[0] + place_xy_offset[0],
        base_spec.place_pos_local[1] + place_xy_offset[1],
    )
    object_pos = object_root_pose_on_support(
        xy_pos=object_xy,
        support_surface_z=base_spec.support_surface_z_local,
        object_bbox_min_z=resolved_bbox_min[2],
        bottom_clearance_m=base_spec.object_bottom_clearance_m,
    )
    if resolved_placement_pos is None:
        place_pos = object_root_pose_on_support(
            xy_pos=place_xy,
            support_surface_z=base_spec.support_surface_z_local,
            object_bbox_min_z=resolved_bbox_min[2],
            bottom_clearance_m=base_spec.object_bottom_clearance_m,
        )
        instruction = instruction_for_object(object_label)
    else:
        place_pos = (
            resolved_placement_pos[0],
            resolved_placement_pos[1],
            object_root_z_on_support(
                support_surface_z=base_spec.support_surface_z_local,
                object_bbox_min_z=resolved_bbox_min[2],
                bottom_clearance_m=base_spec.object_bottom_clearance_m,
            ),
        )
        instruction = instruction_for_object_and_receptacle(object_label, placement_label)

    return PickPlaceTaskSpec(
        instruction=instruction,
        object_name=base_spec.object_name,
        ee_body_name=base_spec.ee_body_name,
        object_pos_local=object_pos,
        place_pos_local=place_pos,
        tcp_offset_local=base_spec.tcp_offset_local,
        grasp_closing_axis_xy=(
            grasp_closing_axis_xy
            if grasp_closing_axis_xy is not None
            else base_spec.grasp_closing_axis_xy
        ),
        object_local_bbox_min=resolved_bbox_min,
        object_local_bbox_max=resolved_bbox_max,
        placement_target_pos_local=resolved_placement_pos,
        placement_target_local_bbox_min=resolved_placement_bbox_min,
        placement_target_local_bbox_max=resolved_placement_bbox_max,
        object_transit_clearance_m=base_spec.object_transit_clearance_m,
        pregrasp_clearance_m=base_spec.pregrasp_clearance_m,
        top_grasp_depth_m=base_spec.top_grasp_depth_m,
        support_surface_z_local=base_spec.support_surface_z_local,
        object_bottom_clearance_m=base_spec.object_bottom_clearance_m,
        place_pregrasp_clearance_m=base_spec.place_pregrasp_clearance_m,
        receptacle_release_bottom_clearance_m=base_spec.receptacle_release_bottom_clearance_m,
        lift_height_m=base_spec.lift_height_m,
        open_finger_m=base_spec.open_finger_m,
        closed_finger_m=base_spec.closed_finger_m,
        free_space_max_speed_m_s=base_spec.free_space_max_speed_m_s,
        approach_max_speed_m_s=base_spec.approach_max_speed_m_s,
        lift_max_speed_m_s=base_spec.lift_max_speed_m_s,
        retreat_max_speed_m_s=base_spec.retreat_max_speed_m_s,
        grasp_dwell_s=base_spec.grasp_dwell_s,
        release_dwell_s=base_spec.release_dwell_s,
    )
