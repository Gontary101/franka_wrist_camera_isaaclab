"""Cartesian motion primitives for scripted policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class MinimumJerkScalarProfile:
    """Minimum-jerk scalar progress profile with zero endpoint velocity and acceleration."""

    duration_s: float

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be positive.")

    def sample(self, elapsed_s: float) -> tuple[float, bool]:
        tau = max(0.0, min(elapsed_s / self.duration_s, 1.0))
        alpha = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
        return alpha, elapsed_s >= self.duration_s


@dataclass(slots=True)
class MinimumJerkPoseMotion:
    """Move between two Cartesian poses with minimum-jerk scalar progress."""

    start_pos_w: torch.Tensor
    goal_pos_w: torch.Tensor
    quat_w: torch.Tensor
    start_time_s: float
    profile: MinimumJerkScalarProfile

    def __post_init__(self) -> None:
        self.start_pos_w = self.start_pos_w.clone()
        self.goal_pos_w = self.goal_pos_w.clone()
        self.quat_w = self.quat_w.clone()

    @classmethod
    def from_speed(
        cls,
        start_pos_w: torch.Tensor,
        goal_pos_w: torch.Tensor,
        quat_w: torch.Tensor,
        start_time_s: float,
        max_speed_m_s: float,
    ) -> "MinimumJerkPoseMotion":
        if max_speed_m_s <= 0.0:
            raise ValueError("max_speed_m_s must be positive.")

        distance_m = float(torch.linalg.norm(goal_pos_w - start_pos_w, dim=-1).max().item())
        duration_s = max(distance_m / max_speed_m_s, 1e-6)

        return cls(
            start_pos_w=start_pos_w,
            goal_pos_w=goal_pos_w,
            quat_w=quat_w,
            start_time_s=start_time_s,
            profile=MinimumJerkScalarProfile(duration_s=duration_s),
        )

    def sample(self, sim_time_s: float) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Sample target pose at simulation time."""
        alpha, done = self.profile.sample(sim_time_s - self.start_time_s)
        pos_w = self.start_pos_w + alpha * (self.goal_pos_w - self.start_pos_w)
        return pos_w, self.quat_w, done
