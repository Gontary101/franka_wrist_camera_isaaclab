"""Basic Cartesian motion primitives for scripted policies."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(slots=True)
class TrapezoidalScalarProfile:
    """One-dimensional triangular/trapezoidal motion profile."""

    distance_m: float
    max_speed_m_s: float
    max_accel_m_s2: float

    accel_time_s: float = 0.0
    cruise_time_s: float = 0.0
    duration_s: float = 0.0
    peak_speed_m_s: float = 0.0
    accel_distance_m: float = 0.0
    cruise_distance_m: float = 0.0

    def __post_init__(self) -> None:
        if self.distance_m < 0.0:
            raise ValueError("distance_m must be non-negative.")
        if self.max_speed_m_s <= 0.0:
            raise ValueError("max_speed_m_s must be positive.")
        if self.max_accel_m_s2 <= 0.0:
            raise ValueError("max_accel_m_s2 must be positive.")

        if self.distance_m == 0.0:
            return

        time_to_max = self.max_speed_m_s / self.max_accel_m_s2
        accel_distance = 0.5 * self.max_accel_m_s2 * time_to_max * time_to_max

        if 2.0 * accel_distance >= self.distance_m:
            self.peak_speed_m_s = math.sqrt(self.distance_m * self.max_accel_m_s2)
            self.accel_time_s = self.peak_speed_m_s / self.max_accel_m_s2
            self.cruise_time_s = 0.0
            self.accel_distance_m = 0.5 * self.distance_m
            self.cruise_distance_m = 0.0
            self.duration_s = 2.0 * self.accel_time_s
        else:
            self.peak_speed_m_s = self.max_speed_m_s
            self.accel_time_s = time_to_max
            self.accel_distance_m = accel_distance
            self.cruise_distance_m = self.distance_m - 2.0 * accel_distance
            self.cruise_time_s = self.cruise_distance_m / self.max_speed_m_s
            self.duration_s = 2.0 * self.accel_time_s + self.cruise_time_s

    def sample(self, elapsed_s: float) -> tuple[float, bool]:
        """Return travelled distance and completion flag."""
        if self.distance_m == 0.0:
            return 0.0, True

        t = max(0.0, min(elapsed_s, self.duration_s))

        if t <= self.accel_time_s:
            travelled = 0.5 * self.max_accel_m_s2 * t * t
        elif t <= self.accel_time_s + self.cruise_time_s:
            cruise_t = t - self.accel_time_s
            travelled = self.accel_distance_m + self.peak_speed_m_s * cruise_t
        else:
            decel_t = t - self.accel_time_s - self.cruise_time_s
            travelled = (
                self.accel_distance_m
                + self.cruise_distance_m
                + self.peak_speed_m_s * decel_t
                - 0.5 * self.max_accel_m_s2 * decel_t * decel_t
            )

        return min(travelled, self.distance_m), elapsed_s >= self.duration_s


@dataclass(slots=True)
class LinearPoseMotion:
    """Move along a straight Cartesian segment using a trapezoidal scalar profile."""

    start_pos_w: torch.Tensor
    goal_pos_w: torch.Tensor
    quat_w: torch.Tensor
    start_time_s: float
    profile: TrapezoidalScalarProfile

    def __post_init__(self) -> None:
        self.start_pos_w = self.start_pos_w.clone()
        self.goal_pos_w = self.goal_pos_w.clone()
        self.quat_w = self.quat_w.clone()

    @classmethod
    def from_limits(
        cls,
        start_pos_w: torch.Tensor,
        goal_pos_w: torch.Tensor,
        quat_w: torch.Tensor,
        start_time_s: float,
        max_speed_m_s: float,
        max_accel_m_s2: float,
    ) -> "LinearPoseMotion":
        distance_m = float(torch.linalg.norm(goal_pos_w - start_pos_w, dim=-1).max().item())
        profile = TrapezoidalScalarProfile(
            distance_m=distance_m,
            max_speed_m_s=max_speed_m_s,
            max_accel_m_s2=max_accel_m_s2,
        )
        return cls(
            start_pos_w=start_pos_w,
            goal_pos_w=goal_pos_w,
            quat_w=quat_w,
            start_time_s=start_time_s,
            profile=profile,
        )

    def sample(self, sim_time_s: float) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Sample target pose at simulation time."""
        travelled_m, done = self.profile.sample(sim_time_s - self.start_time_s)

        if self.profile.distance_m == 0.0:
            alpha = 1.0
        else:
            alpha = travelled_m / self.profile.distance_m

        pos_w = self.start_pos_w + alpha * (self.goal_pos_w - self.start_pos_w)
        return pos_w, self.quat_w, done
