"""Reaching task definitions."""

from __future__ import annotations

from dataclasses import dataclass

from .base import TaskSpec


@dataclass(frozen=True, slots=True)
class ReachingTaskSpec(TaskSpec):
    """Static single-object reaching task."""

    object_name: str = "target_cube"
    ee_body_name: str = "panda_hand"
    instruction: str = "reach the object"

    object_pos_local: tuple[float, float, float] = (0.58, -0.16, 1.08)
    tcp_offset_local: tuple[float, float, float] = (0.0, 0.0, 0.10)

    pregrasp_height_m: float = 0.16
    open_finger_m: float = 0.04

    free_space_max_speed_m_s: float = 0.22
    free_space_max_accel_m_s2: float = 0.45

    approach_max_speed_m_s: float = 0.08
    approach_max_accel_m_s2: float = 0.20

    reach_dwell_s: float = 1.0


def instruction_for_object(object_label: str) -> str:
    return f"reach the {object_label}"


def make_reaching_episode_spec(
    base_spec: ReachingTaskSpec,
    object_xy_offset: tuple[float, float],
    object_label: str,
) -> ReachingTaskSpec:
    object_pos = (
        base_spec.object_pos_local[0] + object_xy_offset[0],
        base_spec.object_pos_local[1] + object_xy_offset[1],
        base_spec.object_pos_local[2],
    )

    return ReachingTaskSpec(
        instruction=instruction_for_object(object_label),
        object_name=base_spec.object_name,
        ee_body_name=base_spec.ee_body_name,
        object_pos_local=object_pos,
        tcp_offset_local=base_spec.tcp_offset_local,
        pregrasp_height_m=base_spec.pregrasp_height_m,
        open_finger_m=base_spec.open_finger_m,
        free_space_max_speed_m_s=base_spec.free_space_max_speed_m_s,
        free_space_max_accel_m_s2=base_spec.free_space_max_accel_m_s2,
        approach_max_speed_m_s=base_spec.approach_max_speed_m_s,
        approach_max_accel_m_s2=base_spec.approach_max_accel_m_s2,
        reach_dwell_s=base_spec.reach_dwell_s,
    )
