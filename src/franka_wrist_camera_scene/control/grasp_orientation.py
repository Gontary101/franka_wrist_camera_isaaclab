"""Grasp orientation helpers."""

from __future__ import annotations

import math

import torch


def yaw_from_planar_axis(axis_xy: tuple[float, float]) -> float:
    """Return world yaw angle for a 2D axis."""
    return math.atan2(float(axis_xy[1]), float(axis_xy[0]))


def yaw_quat_wxyz(yaw_rad: float, device: torch.device) -> torch.Tensor:
    """Quaternion for rotation around world Z."""
    half = 0.5 * yaw_rad
    return torch.tensor(
        [math.cos(half), 0.0, 0.0, math.sin(half)],
        device=device,
        dtype=torch.float32,
    )


def quat_multiply_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two WXYZ quaternions."""
    w1, x1, y1, z1 = q1.unbind()
    w2, x2, y2, z2 = q2.unbind()
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )
    )


def downward_gripper_quat_for_closing_axis(
    closing_axis_xy: tuple[float, float],
    device: torch.device,
) -> torch.Tensor:
    """Return downward gripper orientation with yaw aligned to a planar closing axis."""
    base_down_quat = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device, dtype=torch.float32)
    yaw_quat = yaw_quat_wxyz(yaw_from_planar_axis(closing_axis_xy), device=device)
    return quat_multiply_wxyz(yaw_quat, base_down_quat)
