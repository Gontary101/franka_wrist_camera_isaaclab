"""Basic Cartesian motion primitives for scripted policies."""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(slots=True)
class LinearPoseMotion:
    """Move linearly from one Cartesian pose to another over a fixed duration."""

    start_pos_w: torch.Tensor
    goal_pos_w: torch.Tensor
    quat_w: torch.Tensor
    duration_s: float
    start_time_s: float

    def __post_init__(self) -> None:
        self.start_pos_w = self.start_pos_w.clone()
        self.goal_pos_w = self.goal_pos_w.clone()
        self.quat_w = self.quat_w.clone()

    def sample(self, sim_time_s: float) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Sample target pose at simulation time."""
        if self.duration_s <= 0.0:
            raise ValueError("LinearPoseMotion duration_s must be positive.")

        alpha = (sim_time_s - self.start_time_s) / self.duration_s
        alpha = float(max(0.0, min(1.0, alpha)))

        pos_w = self.start_pos_w + alpha * (self.goal_pos_w - self.start_pos_w)
        done = alpha >= 1.0
        return pos_w, self.quat_w, done
