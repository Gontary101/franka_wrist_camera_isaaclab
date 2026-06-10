"""Trajectory generation utilities for circular and custom target trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch

from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_apply

from ..settings import CIRCLE_CENTER_LOCAL, CIRCLE_DIAMETER_M, CIRCLE_FREQUENCY_HZ, GRIPPER_DOWN_QUAT_WXYZ


@dataclass(frozen=True, slots=True)
class CircleTrajectoryCfg:
    """Circular end-effector path configurations relative to environment origins."""

    center_local: tuple[float, float, float] = CIRCLE_CENTER_LOCAL
    diameter_m: float = CIRCLE_DIAMETER_M
    frequency_hz: float = CIRCLE_FREQUENCY_HZ
    orientation_wxyz: tuple[float, float, float, float] = GRIPPER_DOWN_QUAT_WXYZ
    tcp_offset_local: tuple[float, float, float] = (0.0, 0.0, 0.10)
    preview_points: int = 96

    @property
    def radius_m(self) -> float:
        return 0.5 * self.diameter_m


def circle_pose_w(
    scene: InteractiveScene,
    sim_time_s: float,
    cfg: CircleTrajectoryCfg,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate the target wrist pose in world frame to trace a circle in local frame."""
    phase = 2.0 * math.pi * cfg.frequency_hz * sim_time_s
    center_local = torch.tensor(cfg.center_local, device=device).view(1, 3)
    target_quat_w = torch.tensor(cfg.orientation_wxyz, device=device).view(1, 4)
    tcp_offset_local = torch.tensor(cfg.tcp_offset_local, device=device).view(1, 3)

    tcp_pos_w = center_local.repeat(scene.num_envs, 1) + scene.env_origins
    tcp_pos_w[:, 0] += cfg.radius_m * math.cos(phase)
    tcp_pos_w[:, 1] += cfg.radius_m * math.sin(phase)

    quat_w = target_quat_w.repeat(scene.num_envs, 1)
    tcp_offset_w = quat_apply(quat_w, tcp_offset_local.repeat(scene.num_envs, 1))
    hand_pos_w = tcp_pos_w - tcp_offset_w

    return hand_pos_w, quat_w


def circle_points_w(
    scene: InteractiveScene,
    cfg: CircleTrajectoryCfg,
    device: str | torch.device,
) -> torch.Tensor:
    """Generate preview path points in world coordinates for visualization markers."""
    center_local = torch.tensor(cfg.center_local, device=device).view(1, 3)
    angles = torch.linspace(0.0, 2.0 * math.pi, cfg.preview_points + 1, device=device)[:-1]
    
    points = center_local.repeat(cfg.preview_points, 1)
    points[:, 0] += cfg.radius_m * torch.cos(angles)
    points[:, 1] += cfg.radius_m * torch.sin(angles)

    points_w = points.unsqueeze(0) + scene.env_origins.unsqueeze(1)
    return points_w.reshape(-1, 3)
