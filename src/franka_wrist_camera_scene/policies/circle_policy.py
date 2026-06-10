"""Scripted reaching policies for tracking specified task trajectories."""

from __future__ import annotations

import torch
from isaaclab.scene import InteractiveScene
from isaaclab.assets import Articulation

from ..control.trajectory import CircleTrajectoryCfg, circle_pose_w


class CircleMotionPolicy:
    """Policy that generates target poses to trace a circular end-effector trajectory."""

    def __init__(self, cfg: CircleTrajectoryCfg, gripper_width: float = 0.035):
        self.cfg = cfg
        self.gripper_width = gripper_width
        self._scene = None
        self._device = None

    def bind(self, scene: InteractiveScene, robot: Articulation) -> None:
        """Bind simulation scene and get device reference."""
        self._scene = scene
        self._device = robot.device

    def step(self, obs: dict | None, sim_time_s: float) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Compute the next target end-effector pose and gripper width."""
        target_pos_w, target_quat_w = circle_pose_w(self._scene, sim_time_s, self.cfg, self._device)
        return target_pos_w, target_quat_w, self.gripper_width
