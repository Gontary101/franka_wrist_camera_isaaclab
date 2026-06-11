"""Pick-and-place task definitions."""

from __future__ import annotations

from dataclasses import dataclass
from .base import TaskSpec


@dataclass(frozen=True, slots=True)
class PickPlaceTaskSpec(TaskSpec):
    """Static single-object pick-and-place task."""

    object_name: str = "target_cube"
    ee_body_name: str = "panda_hand"
    instruction: str = "pick up the red cube and place it on the target area"

    object_pos_local: tuple[float, float, float] = (0.58, -0.16, 1.08)
    place_pos_local: tuple[float, float, float] = (0.55, 0.22, 1.08)

    pregrasp_height_m: float = 0.16
    lift_height_m: float = 0.20
    open_finger_m: float = 0.035
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


def instruction_for_color(object_color_name: str) -> str:
    return f"pick up the {object_color_name} cube and place it on the target area"


def make_pick_place_episode_spec(
    base_spec: PickPlaceTaskSpec,
    object_xy_offset: tuple[float, float],
    place_xy_offset: tuple[float, float],
    object_color_name: str,
) -> PickPlaceTaskSpec:
    object_pos = (
        base_spec.object_pos_local[0] + object_xy_offset[0],
        base_spec.object_pos_local[1] + object_xy_offset[1],
        base_spec.object_pos_local[2],
    )
    place_pos = (
        base_spec.place_pos_local[0] + place_xy_offset[0],
        base_spec.place_pos_local[1] + place_xy_offset[1],
        base_spec.place_pos_local[2],
    )

    return PickPlaceTaskSpec(
        instruction=instruction_for_color(object_color_name),
        object_name=base_spec.object_name,
        ee_body_name=base_spec.ee_body_name,
        object_pos_local=object_pos,
        place_pos_local=place_pos,
        pregrasp_height_m=base_spec.pregrasp_height_m,
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

