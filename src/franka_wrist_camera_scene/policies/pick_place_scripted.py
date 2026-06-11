"""Scripted pick-and-place policy using a simple finite-state machine."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_apply

from ..control.motion_primitives import LinearPoseMotion
from ..tasks.pick_place import PickPlaceTaskSpec
from .scripted_base import PolicyCommand


class PickPlaceScriptedPolicy:
    """Scripted finite-state machine policy for deterministic pick-and-place."""

    def __init__(self, spec: PickPlaceTaskSpec):
        self.spec = spec
        self.state = "move_to_pregrasp"
        self._scene = None
        self._device = None
        self._motion = None
        self._state_start_time = None
        self._ee_body_id = None

        # Gripper orientation (always pointing down)
        self.quat_wxyz = torch.tensor([0.0, 1.0, 0.0, 0.0])

    def bind(self, scene: InteractiveScene, robot: Articulation) -> None:
        """Bind simulation scene and get device reference."""
        if scene.num_envs != 1:
            raise RuntimeError("PickPlaceScriptedPolicy currently supports only num_envs=1.")
        self._scene = scene
        self._device = robot.device
        self.quat_wxyz = self.quat_wxyz.to(self._device)
        self._ee_body_id = robot.find_bodies(self.spec.ee_body_name)[0][0]

    def reset(self) -> None:
        """Reset the policy to the initial state."""
        self.state = "move_to_pregrasp"
        self._motion = None
        self._state_start_time = None

    def step(self, obs: dict | None, sim_time_s: float) -> PolicyCommand:
        """Compute the next command target according to the FSM state."""
        if self._scene is None or self._device is None or self._ee_body_id is None:
            raise RuntimeError("PickPlaceScriptedPolicy was not bound before step().")

        robot = self._scene["robot"]
        ee_pos_w = robot.data.body_pose_w[:, self._ee_body_id, :3]  # shape: (num_envs, 3)
        num_envs = self._scene.num_envs

        # Target definitions (TCP targets)
        # Dynamic object position from the simulated RigidObject (allows randomization)
        obj_pos = self._scene[self.spec.object_name].data.root_pos_w  # shape: (num_envs, 3)

        place_local = torch.tensor(self.spec.place_pos_local, device=self._device)
        # Convert env-local coordinates to world coordinates using env origins
        place_pos = self._scene.env_origins + place_local.view(1, 3)

        # Subtract TCP offset (0.10m down in local coordinates) to get the hand position targets
        tcp_offset_local = torch.tensor([0.0, 0.0, 0.10], device=self._device).view(1, 3)
        tcp_offset_w = quat_apply(self.quat_wxyz.view(1, 4), tcp_offset_local).view(3)

        obj_hand_pos = obj_pos - tcp_offset_w.view(1, 3)
        place_hand_pos = place_pos - tcp_offset_w.view(1, 3)

        pregrasp_pos = obj_hand_pos.clone()
        pregrasp_pos[:, 2] += self.spec.pregrasp_height_m

        lift_pos = obj_hand_pos.clone()
        lift_pos[:, 2] += self.spec.lift_height_m

        place_pre_pos = place_hand_pos.clone()
        place_pre_pos[:, 2] += self.spec.lift_height_m

        target_pos_w = ee_pos_w.clone()
        target_quat_w = self.quat_wxyz.repeat(num_envs, 1)
        finger_opening = self.spec.open_finger_m
        done = False

        if self.state == "move_to_pregrasp":
            if self._motion is None:
                self._motion = LinearPoseMotion.from_limits(
                    start_pos_w=ee_pos_w,
                    goal_pos_w=pregrasp_pos,
                    quat_w=target_quat_w,
                    start_time_s=sim_time_s,
                    max_speed_m_s=self.spec.free_space_max_speed_m_s,
                    max_accel_m_s2=self.spec.free_space_max_accel_m_s2,
                )
            pos, quat, finished = self._motion.sample(sim_time_s)
            target_pos_w = pos
            target_quat_w = quat
            if finished:
                self.state = "move_to_grasp"
                self._motion = None

        elif self.state == "move_to_grasp":
            if self._motion is None:
                self._motion = LinearPoseMotion.from_limits(
                    start_pos_w=ee_pos_w,
                    goal_pos_w=obj_hand_pos,
                    quat_w=target_quat_w,
                    start_time_s=sim_time_s,
                    max_speed_m_s=self.spec.approach_max_speed_m_s,
                    max_accel_m_s2=self.spec.approach_max_accel_m_s2,
                )
            pos, quat, finished = self._motion.sample(sim_time_s)
            target_pos_w = pos
            target_quat_w = quat
            if finished:
                self.state = "close"
                self._state_start_time = sim_time_s
                self._motion = None

        elif self.state == "close":
            target_pos_w = obj_hand_pos
            finger_opening = self.spec.closed_finger_m
            if sim_time_s - self._state_start_time >= self.spec.grasp_dwell_s:
                self.state = "lift"
                self._state_start_time = None

        elif self.state == "lift":
            finger_opening = self.spec.closed_finger_m
            if self._motion is None:
                self._motion = LinearPoseMotion.from_limits(
                    start_pos_w=ee_pos_w,
                    goal_pos_w=lift_pos,
                    quat_w=target_quat_w,
                    start_time_s=sim_time_s,
                    max_speed_m_s=self.spec.lift_max_speed_m_s,
                    max_accel_m_s2=self.spec.lift_max_accel_m_s2,
                )
            pos, quat, finished = self._motion.sample(sim_time_s)
            target_pos_w = pos
            target_quat_w = quat
            if finished:
                self.state = "move_to_place"
                self._motion = None

        elif self.state == "move_to_place":
            finger_opening = self.spec.closed_finger_m
            if self._motion is None:
                self._motion = LinearPoseMotion.from_limits(
                    start_pos_w=ee_pos_w,
                    goal_pos_w=place_pre_pos,
                    quat_w=target_quat_w,
                    start_time_s=sim_time_s,
                    max_speed_m_s=self.spec.free_space_max_speed_m_s,
                    max_accel_m_s2=self.spec.free_space_max_accel_m_s2,
                )
            pos, quat, finished = self._motion.sample(sim_time_s)
            target_pos_w = pos
            target_quat_w = quat
            if finished:
                self.state = "lower"
                self._motion = None

        elif self.state == "lower":
            finger_opening = self.spec.closed_finger_m
            if self._motion is None:
                self._motion = LinearPoseMotion.from_limits(
                    start_pos_w=ee_pos_w,
                    goal_pos_w=place_hand_pos,
                    quat_w=target_quat_w,
                    start_time_s=sim_time_s,
                    max_speed_m_s=self.spec.approach_max_speed_m_s,
                    max_accel_m_s2=self.spec.approach_max_accel_m_s2,
                )
            pos, quat, finished = self._motion.sample(sim_time_s)
            target_pos_w = pos
            target_quat_w = quat
            if finished:
                self.state = "open"
                self._state_start_time = sim_time_s
                self._motion = None

        elif self.state == "open":
            target_pos_w = place_hand_pos
            finger_opening = self.spec.open_finger_m
            if sim_time_s - self._state_start_time >= self.spec.release_dwell_s:
                self.state = "retreat"
                self._state_start_time = None

        elif self.state == "retreat":
            finger_opening = self.spec.open_finger_m
            if self._motion is None:
                self._motion = LinearPoseMotion.from_limits(
                    start_pos_w=ee_pos_w,
                    goal_pos_w=place_pre_pos,
                    quat_w=target_quat_w,
                    start_time_s=sim_time_s,
                    max_speed_m_s=self.spec.retreat_max_speed_m_s,
                    max_accel_m_s2=self.spec.retreat_max_accel_m_s2,
                )
            pos, quat, finished = self._motion.sample(sim_time_s)
            target_pos_w = pos
            target_quat_w = quat
            if finished:
                self.state = "done"
                self._motion = None

        elif self.state == "done":
            target_pos_w = place_pre_pos
            finger_opening = self.spec.open_finger_m
            done = True

        return PolicyCommand(
            target_pos_w=target_pos_w,
            target_quat_w=target_quat_w,
            finger_opening_m=finger_opening,
            done=done,
        )
