"""Gripper controller interface for joint position control of fingers."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene


class GripperController:
    """Robot gripper controller for setting parallel finger widths."""

    def __init__(self, finger_joint_expr: str = "panda_finger_joint.*"):
        self.finger_joint_expr = finger_joint_expr
        self._finger_joint_ids = None
        self._target_width = None

    def bind(self, scene: InteractiveScene, robot: Articulation) -> None:
        """Resolve finger joint indices and initialize target buffer."""
        self._finger_joint_ids, _ = robot.find_joints(self.finger_joint_expr)
        self._target_width = torch.zeros(
            (scene.num_envs, len(self._finger_joint_ids)),
            device=robot.device,
        )

    def set_width(self, width: float | torch.Tensor) -> None:
        """Set target gripper width."""
        if self._target_width is None:
            raise RuntimeError("GripperController was not bound before set_width().")
        if isinstance(width, (float, int)):
            self._target_width.fill_(width)
        else:
            self._target_width[:] = width

    def apply(self, robot: Articulation) -> None:
        """Apply finger width targets to the robot simulator."""
        if self._finger_joint_ids is None or self._target_width is None:
            raise RuntimeError("GripperController was not bound before apply().")
        robot.set_joint_position_target(self._target_width, joint_ids=self._finger_joint_ids)
