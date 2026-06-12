"""Cartesian motion primitives for scripted policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch


MINIMUM_JERK_PEAK_VELOCITY_SCALE = 1.875
MINIMUM_MOTION_DURATION_S = 1e-6
ZERO_LENGTH_SEGMENT_EPS = 1e-8


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
        duration_s = max(
            MINIMUM_JERK_PEAK_VELOCITY_SCALE * distance_m / max_speed_m_s,
            MINIMUM_MOTION_DURATION_S,
        )

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


@dataclass(slots=True)
class MinimumJerkWaypointMotion:
    """Move through Cartesian waypoints with one continuous minimum-jerk progress profile."""

    waypoints_w: tuple[torch.Tensor, ...]
    quat_w: torch.Tensor
    start_time_s: float
    profile: MinimumJerkScalarProfile
    segment_lengths: torch.Tensor
    cumulative_scaled_lengths: torch.Tensor

    @classmethod
    def from_speed(
        cls,
        waypoints_w: tuple[torch.Tensor, ...],
        quat_w: torch.Tensor,
        start_time_s: float,
        max_speed_m_s: float,
    ) -> "MinimumJerkWaypointMotion":
        return cls.from_segment_speeds(
            waypoints_w=waypoints_w,
            quat_w=quat_w,
            start_time_s=start_time_s,
            max_speed_m_s=(max_speed_m_s,) * (len(waypoints_w) - 1),
        )

    @classmethod
    def from_segment_speeds(
        cls,
        waypoints_w: tuple[torch.Tensor, ...],
        quat_w: torch.Tensor,
        start_time_s: float,
        max_speed_m_s: tuple[float, ...],
    ) -> "MinimumJerkWaypointMotion":
        if len(waypoints_w) < 2:
            raise ValueError("MinimumJerkWaypointMotion requires at least two waypoints.")
        if len(max_speed_m_s) != len(waypoints_w) - 1:
            raise ValueError("MinimumJerkWaypointMotion requires one speed per segment.")
        if any(speed <= 0.0 for speed in max_speed_m_s):
            raise ValueError("max_speed_m_s values must be positive.")

        cloned = tuple(point.clone() for point in waypoints_w)
        segment_lengths = torch.stack(
            [
                torch.linalg.norm(cloned[index + 1] - cloned[index], dim=-1).max()
                for index in range(len(cloned) - 1)
            ]
        )
        speed_limits = torch.tensor(
            max_speed_m_s,
            device=segment_lengths.device,
            dtype=segment_lengths.dtype,
        )
        scaled_lengths = segment_lengths / speed_limits
        total_scaled_length = float(scaled_lengths.sum().item())
        duration_s = max(
            MINIMUM_JERK_PEAK_VELOCITY_SCALE * total_scaled_length,
            MINIMUM_MOTION_DURATION_S,
        )
        cumulative_scaled_lengths = torch.cat(
            [
                torch.zeros(1, device=segment_lengths.device, dtype=segment_lengths.dtype),
                torch.cumsum(scaled_lengths, dim=0),
            ]
        )

        return cls(
            waypoints_w=cloned,
            quat_w=quat_w.clone(),
            start_time_s=start_time_s,
            profile=MinimumJerkScalarProfile(duration_s=duration_s),
            segment_lengths=segment_lengths,
            cumulative_scaled_lengths=cumulative_scaled_lengths,
        )

    def sample(self, sim_time_s: float) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Sample target pose at simulation time."""
        alpha, done = self.profile.sample(sim_time_s - self.start_time_s)
        total_scaled_length = self.cumulative_scaled_lengths[-1]
        target_scaled_length = alpha * total_scaled_length

        segment_index = int(
            torch.searchsorted(
                self.cumulative_scaled_lengths,
                target_scaled_length,
                right=True,
            ).item()
            - 1
        )
        segment_index = max(0, min(segment_index, len(self.waypoints_w) - 2))

        segment_start = self.cumulative_scaled_lengths[segment_index]
        segment_length = self.cumulative_scaled_lengths[segment_index + 1] - segment_start
        if float(segment_length.item()) <= ZERO_LENGTH_SEGMENT_EPS:
            local_alpha = 1.0
        else:
            local_alpha = float(((target_scaled_length - segment_start) / segment_length).item())

        start = self.waypoints_w[segment_index]
        goal = self.waypoints_w[segment_index + 1]
        pos_w = start + local_alpha * (goal - start)
        return pos_w, self.quat_w, done
