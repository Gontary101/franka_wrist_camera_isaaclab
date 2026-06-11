"""Pick-and-place task definitions."""

from __future__ import annotations

from dataclasses import dataclass
from .base import TaskSpec


@dataclass(frozen=True, slots=True)
class PickPlaceTaskSpec(TaskSpec):
    """Static single-object pick-and-place task."""

    object_name: str = "target_cube"
    target_name: str = "place_target"
    ee_body_name: str = "panda_hand"
    instruction: str = "pick up the red cube and place it on the target area"

    object_pos_local: tuple[float, float, float] = (0.58, -0.16, 1.08)
    place_pos_local: tuple[float, float, float] = (0.55, 0.22, 1.08)

    pregrasp_height_m: float = 0.16
    lift_height_m: float = 0.20
    open_finger_m: float = 0.035
    closed_finger_m: float = 0.0


def make_pick_place_episode_spec(
    base_spec: PickPlaceTaskSpec,
    episode_id: int,
    object_xy_offset: tuple[float, float],
    place_xy_offset: tuple[float, float],
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
        instruction=base_spec.instruction,
        object_name=base_spec.object_name,
        target_name=base_spec.target_name,
        ee_body_name=base_spec.ee_body_name,
        object_pos_local=object_pos,
        place_pos_local=place_pos,
        pregrasp_height_m=base_spec.pregrasp_height_m,
        lift_height_m=base_spec.lift_height_m,
        open_finger_m=base_spec.open_finger_m,
        closed_finger_m=base_spec.closed_finger_m,
    )

