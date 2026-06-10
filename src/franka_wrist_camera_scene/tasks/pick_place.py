"""Pick-and-place task definitions."""

from __future__ import annotations

from dataclasses import dataclass

from .base import TaskSpec


@dataclass(frozen=True, slots=True)
class PickPlaceTaskSpec(TaskSpec):
    """Static single-object pick-and-place task."""

    object_name: str = "target_cube"
    target_name: str = "bin"
    instruction: str = "pick up the red cube and place it in the bin"

    object_pos_w: tuple[float, float, float] = (0.58, -0.16, 1.0875)
    place_pos_w: tuple[float, float, float] = (0.55, 0.22, 1.0875)

    pregrasp_height_m: float = 0.16
    lift_height_m: float = 0.20
    open_finger_m: float = 0.035
    closed_finger_m: float = 0.0
