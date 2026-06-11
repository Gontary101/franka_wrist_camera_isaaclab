"""Episode recorder for internal raw dataset format."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from isaaclab.scene import InteractiveScene

from franka_wrist_camera_scene.policies.scripted_base import PolicyCommand
from franka_wrist_camera_scene.episode.schema import EpisodeMetadata


@dataclass(slots=True)
class EpisodeRecorder:
    """Record one episode into a simple internal directory format."""

    output_dir: Path
    episode_id: int
    task_name: str
    instruction: str
    sim_dt: float
    ee_body_id: int
    object_name: str

    record_cameras: bool = False
    record_depth: bool = False

    joint_pos: list[np.ndarray] = field(default_factory=list)
    joint_vel: list[np.ndarray] = field(default_factory=list)
    ee_pos_w: list[np.ndarray] = field(default_factory=list)
    object_pos_w: list[np.ndarray] = field(default_factory=list)
    action_target_pos_w: list[np.ndarray] = field(default_factory=list)
    action_target_quat_w: list[np.ndarray] = field(default_factory=list)
    action_finger_opening_m: list[float] = field(default_factory=list)

    timestamps_s: list[float] = field(default_factory=list)
    camera_step_indices: list[int] = field(default_factory=list)
    camera_timestamps_s: list[float] = field(default_factory=list)
    agent_rgb: list[np.ndarray] = field(default_factory=list)
    wrist_rgb: list[np.ndarray] = field(default_factory=list)
    agent_depth: list[np.ndarray] = field(default_factory=list)
    wrist_depth: list[np.ndarray] = field(default_factory=list)

    @property
    def episode_dir(self) -> Path:
        return self.output_dir / f"{self.episode_id:06d}"

    def validate_output_path(self) -> None:
        if self.episode_dir.exists():
            raise FileExistsError(f"Episode directory already exists: {self.episode_dir}")

    def record_step(self, scene: InteractiveScene, cmd: PolicyCommand, step: int, sim_time_s: float) -> None:
        # Dataset convention: record state_t and command_t before advancing to state_{t+1}.
        robot = scene["robot"]
        obj = scene[self.object_name]

        self.timestamps_s.append(float(sim_time_s))

        self.joint_pos.append(robot.data.joint_pos.detach().cpu().numpy().copy())
        self.joint_vel.append(robot.data.joint_vel.detach().cpu().numpy().copy())
        self.ee_pos_w.append(robot.data.body_pose_w[:, self.ee_body_id, :3].detach().cpu().numpy().copy())
        self.object_pos_w.append(obj.data.root_pos_w.detach().cpu().numpy().copy())

        self.action_target_pos_w.append(cmd.target_pos_w.detach().cpu().numpy().copy())
        self.action_target_quat_w.append(cmd.target_quat_w.detach().cpu().numpy().copy())
        self.action_finger_opening_m.append(float(cmd.finger_opening_m))

    def record_cameras_step(self, scene: InteractiveScene, step: int, sim_time_s: float) -> None:
        """Record camera observations for this control step."""
        if not self.record_cameras:
            return

        self.camera_step_indices.append(int(step))
        self.camera_timestamps_s.append(float(sim_time_s))

        for camera_name, buffer in (
            ("agent_camera", self.agent_rgb),
            ("wrist_camera", self.wrist_rgb),
        ):
            rgb = scene[camera_name].data.output["rgb"][0].detach().cpu().numpy()[..., :3]
            buffer.append(np.clip(rgb, 0, 255).astype(np.uint8).copy())

        if self.record_depth:
            for camera_name, buffer in (
                ("agent_camera", self.agent_depth),
                ("wrist_camera", self.wrist_depth),
            ):
                depth = scene[camera_name].data.output["distance_to_image_plane"][0, ..., 0]
                buffer.append(depth.detach().cpu().numpy().astype(np.float32).copy())

    def save(self, success: bool) -> Path:
        episode_dir = self.episode_dir
        if episode_dir.exists():
            raise FileExistsError(f"Episode directory already exists: {episode_dir}")
        episode_dir.mkdir(parents=True)

        arrays = {
            "timestamps_s": np.asarray(self.timestamps_s, dtype=np.float32),
            "joint_pos": np.asarray(self.joint_pos),
            "joint_vel": np.asarray(self.joint_vel),
            "ee_pos_w": np.asarray(self.ee_pos_w),
            "object_pos_w": np.asarray(self.object_pos_w),
            "action_target_pos_w": np.asarray(self.action_target_pos_w),
            "action_target_quat_w": np.asarray(self.action_target_quat_w),
            "action_finger_opening_m": np.asarray(self.action_finger_opening_m),
        }

        if self.record_cameras:
            arrays.update(
                camera_step_indices=np.asarray(self.camera_step_indices, dtype=np.int64),
                camera_timestamps_s=np.asarray(self.camera_timestamps_s, dtype=np.float32),
                agent_rgb=np.asarray(self.agent_rgb, dtype=np.uint8),
                wrist_rgb=np.asarray(self.wrist_rgb, dtype=np.uint8),
            )

        if self.record_cameras and self.record_depth:
            arrays.update(
                agent_depth=np.asarray(self.agent_depth, dtype=np.float32),
                wrist_depth=np.asarray(self.wrist_depth, dtype=np.float32),
            )

        np.savez_compressed(episode_dir / "trajectory.npz", **arrays)

        meta = EpisodeMetadata(
            episode_id=self.episode_id,
            task_name=self.task_name,
            instruction=self.instruction,
            success=success,
            num_steps=len(self.joint_pos),
            sim_dt=self.sim_dt,
            record_cameras=self.record_cameras,
            record_depth=self.record_depth,
            num_camera_frames=len(self.camera_step_indices) if self.record_cameras else 0,
        )
        meta.save(episode_dir / "meta.json")
        return episode_dir


