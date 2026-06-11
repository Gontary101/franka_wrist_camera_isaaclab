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

    joint_pos: list[np.ndarray] = field(default_factory=list)
    joint_vel: list[np.ndarray] = field(default_factory=list)
    ee_pos_w: list[np.ndarray] = field(default_factory=list)
    object_pos_w: list[np.ndarray] = field(default_factory=list)
    action_target_pos_w: list[np.ndarray] = field(default_factory=list)
    action_target_quat_w: list[np.ndarray] = field(default_factory=list)
    action_finger_opening_m: list[float] = field(default_factory=list)

    def record_step(self, scene: InteractiveScene, cmd: PolicyCommand) -> None:
        robot = scene["robot"]
        obj = scene[self.object_name]

        self.joint_pos.append(robot.data.joint_pos.detach().cpu().numpy().copy())
        self.joint_vel.append(robot.data.joint_vel.detach().cpu().numpy().copy())
        self.ee_pos_w.append(robot.data.body_pose_w[:, self.ee_body_id, :3].detach().cpu().numpy().copy())
        self.object_pos_w.append(obj.data.root_pos_w.detach().cpu().numpy().copy())

        self.action_target_pos_w.append(cmd.target_pos_w.detach().cpu().numpy().copy())
        self.action_target_quat_w.append(cmd.target_quat_w.detach().cpu().numpy().copy())
        self.action_finger_opening_m.append(float(cmd.finger_opening_m))

    def save(self, success: bool) -> Path:
        episode_dir = self.output_dir / f"{self.episode_id:06d}"
        episode_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            episode_dir / "trajectory.npz",
            joint_pos=np.asarray(self.joint_pos),
            joint_vel=np.asarray(self.joint_vel),
            ee_pos_w=np.asarray(self.ee_pos_w),
            object_pos_w=np.asarray(self.object_pos_w),
            action_target_pos_w=np.asarray(self.action_target_pos_w),
            action_target_quat_w=np.asarray(self.action_target_quat_w),
            action_finger_opening_m=np.asarray(self.action_finger_opening_m),
        )

        meta = EpisodeMetadata(
            episode_id=self.episode_id,
            task_name=self.task_name,
            instruction=self.instruction,
            success=success,
            num_steps=len(self.joint_pos),
            sim_dt=self.sim_dt,
        )
        meta.save(episode_dir / "meta.json")
        return episode_dir
