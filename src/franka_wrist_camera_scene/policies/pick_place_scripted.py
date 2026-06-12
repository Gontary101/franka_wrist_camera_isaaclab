"""Scripted pick-and-place policy using a simple finite-state machine."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_apply

from ..control.grasp_orientation import downward_gripper_quat_for_closing_axis
from ..control.motion_primitives import MinimumJerkPoseMotion
from ..tasks.placement_geometry import object_root_z_on_support
from ..tasks.pick_place import PickPlaceTaskSpec
from .scripted_base import PolicyCommand


class PickPlaceScriptedPolicy:
    """Scripted finite-state machine policy for deterministic pick-and-place."""

    def __init__(self, spec: PickPlaceTaskSpec):
        self.spec = spec
        self.state = "move_to_object_transit"
        self._scene = None
        self._device = None
        self._motion = None
        self._state_start_time = None
        self._ee_body_id = None
        self._grasp_tcp_offset_from_root_w = None
        self._lift_pos_w = None

        self.quat_wxyz = torch.tensor([0.0, 1.0, 0.0, 0.0])

    def _actual_tcp_pos_w(self, ee_pos_w: torch.Tensor, tcp_offset_w: torch.Tensor) -> torch.Tensor:
        return ee_pos_w + tcp_offset_w.view(1, 3)

    def _object_root_on_support_w(self, xy_pos_w: torch.Tensor) -> torch.Tensor:
        if self.spec.object_local_bbox_min is None:
            raise RuntimeError("Pick-place requires object bbox metadata for placement height.")

        root_pos = xy_pos_w.clone()
        root_pos[:, 2] = object_root_z_on_support(
            support_surface_z=self.spec.support_surface_z_local,
            object_bbox_min_z=float(self.spec.object_local_bbox_min[2]),
            bottom_clearance_m=self.spec.object_bottom_clearance_m,
        )
        return root_pos

    def _object_top_tcp_targets_w(self, obj_pos_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.spec.object_local_bbox_min is None or self.spec.object_local_bbox_max is None:
            raise RuntimeError("Pick-place requires object bbox metadata for top grasp targeting.")

        bbox_max_z = float(self.spec.object_local_bbox_max[2])

        grasp_tcp = obj_pos_w.clone()
        grasp_tcp[:, 2] = obj_pos_w[:, 2] + bbox_max_z - self.spec.top_grasp_depth_m

        pregrasp_tcp = grasp_tcp.clone()
        pregrasp_tcp[:, 2] += self.spec.pregrasp_clearance_m

        transit_tcp = grasp_tcp.clone()
        transit_tcp[:, 2] += self.spec.object_transit_clearance_m

        return grasp_tcp, pregrasp_tcp, transit_tcp

    def bind(self, scene: InteractiveScene, robot: Articulation) -> None:
        """Bind simulation scene and get device reference."""
        if scene.num_envs != 1:
            raise RuntimeError("PickPlaceScriptedPolicy currently supports only num_envs=1.")
        self._scene = scene
        self._device = robot.device
        if self.spec.grasp_closing_axis_xy is None:
            self.quat_wxyz = self.quat_wxyz.to(self._device)
        else:
            self.quat_wxyz = downward_gripper_quat_for_closing_axis(
                closing_axis_xy=self.spec.grasp_closing_axis_xy,
                device=self._device,
            )
        self._ee_body_id = robot.find_bodies(self.spec.ee_body_name)[0][0]

    def reset(self) -> None:
        """Reset the policy to the initial state."""
        self.state = "move_to_object_transit"
        self._motion = None
        self._state_start_time = None
        self._grasp_tcp_offset_from_root_w = None
        self._lift_pos_w = None

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

        tcp_offset_local = torch.tensor(self.spec.tcp_offset_local, device=self._device).view(1, 3)
        tcp_offset_w = quat_apply(self.quat_wxyz.view(1, 4), tcp_offset_local).view(3)

        obj_grasp_tcp, obj_pregrasp_tcp, obj_transit_tcp = self._object_top_tcp_targets_w(obj_pos)

        obj_hand_pos = obj_grasp_tcp - tcp_offset_w.view(1, 3)
        pregrasp_pos = obj_pregrasp_tcp - tcp_offset_w.view(1, 3)
        object_transit_pos = obj_transit_tcp - tcp_offset_w.view(1, 3)

        lift_pos = None
        if self._lift_pos_w is not None:
            lift_pos = self._lift_pos_w
        else:
            lift_pos = obj_hand_pos.clone()
            lift_pos[:, 2] += self.spec.lift_height_m

        place_hand_pos = None
        place_pre_pos = None
        place_transit_pos = None

        if self._grasp_tcp_offset_from_root_w is not None:
            place_root_pos = self._object_root_on_support_w(place_pos)
            place_release_tcp = place_root_pos + self._grasp_tcp_offset_from_root_w

            place_pre_tcp = place_release_tcp.clone()
            place_pre_tcp[:, 2] += self.spec.place_pregrasp_clearance_m

            place_hand_pos = place_release_tcp - tcp_offset_w.view(1, 3)
            place_pre_pos = place_pre_tcp - tcp_offset_w.view(1, 3)

            place_transit_pos = place_pre_pos.clone()
            place_transit_pos[:, 2] = lift_pos[:, 2]

        if self.state in ["move_to_place_transit", "move_to_place", "lower", "open", "retreat", "done"]:
            if place_hand_pos is None or place_pre_pos is None or place_transit_pos is None or lift_pos is None:
                raise RuntimeError("Placement targets requested before grasp offset was latched.")

        target_pos_w = ee_pos_w.clone()
        target_quat_w = self.quat_wxyz.repeat(num_envs, 1)
        finger_opening = self.spec.open_finger_m
        done = False

        if self.state == "move_to_object_transit":
            if self._motion is None:
                self._motion = MinimumJerkPoseMotion.from_speed(
                    start_pos_w=ee_pos_w,
                    goal_pos_w=object_transit_pos,
                    quat_w=target_quat_w,
                    start_time_s=sim_time_s,
                    max_speed_m_s=self.spec.free_space_max_speed_m_s,
                )
            pos, quat, finished = self._motion.sample(sim_time_s)
            target_pos_w = pos
            target_quat_w = quat
            if finished:
                self.state = "move_to_pregrasp"
                self._motion = None

        elif self.state == "move_to_pregrasp":
            if self._motion is None:
                self._motion = MinimumJerkPoseMotion.from_speed(
                    start_pos_w=ee_pos_w,
                    goal_pos_w=pregrasp_pos,
                    quat_w=target_quat_w,
                    start_time_s=sim_time_s,
                    max_speed_m_s=self.spec.approach_max_speed_m_s,
                )
            pos, quat, finished = self._motion.sample(sim_time_s)
            target_pos_w = pos
            target_quat_w = quat
            if finished:
                self.state = "move_to_grasp"
                self._motion = None

        elif self.state == "move_to_grasp":
            if self._motion is None:
                self._motion = MinimumJerkPoseMotion.from_speed(
                    start_pos_w=ee_pos_w,
                    goal_pos_w=obj_hand_pos,
                    quat_w=target_quat_w,
                    start_time_s=sim_time_s,
                    max_speed_m_s=self.spec.approach_max_speed_m_s,
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
                actual_tcp_pos_w = self._actual_tcp_pos_w(ee_pos_w, tcp_offset_w)
                self._grasp_tcp_offset_from_root_w = (actual_tcp_pos_w - obj_pos).clone()
                self._lift_pos_w = obj_hand_pos.clone()
                self._lift_pos_w[:, 2] += self.spec.lift_height_m
                self.state = "lift"
                self._state_start_time = None

        elif self.state == "lift":
            finger_opening = self.spec.closed_finger_m
            if self._motion is None:
                self._motion = MinimumJerkPoseMotion.from_speed(
                    start_pos_w=ee_pos_w,
                    goal_pos_w=lift_pos,
                    quat_w=target_quat_w,
                    start_time_s=sim_time_s,
                    max_speed_m_s=self.spec.lift_max_speed_m_s,
                )
            pos, quat, finished = self._motion.sample(sim_time_s)
            target_pos_w = pos
            target_quat_w = quat
            if finished:
                self.state = "move_to_place_transit"
                self._motion = None

        elif self.state == "move_to_place_transit":
            finger_opening = self.spec.closed_finger_m
            if self._motion is None:
                self._motion = MinimumJerkPoseMotion.from_speed(
                    start_pos_w=ee_pos_w,
                    goal_pos_w=place_transit_pos,
                    quat_w=target_quat_w,
                    start_time_s=sim_time_s,
                    max_speed_m_s=self.spec.free_space_max_speed_m_s,
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
                self._motion = MinimumJerkPoseMotion.from_speed(
                    start_pos_w=ee_pos_w,
                    goal_pos_w=place_pre_pos,
                    quat_w=target_quat_w,
                    start_time_s=sim_time_s,
                    max_speed_m_s=self.spec.approach_max_speed_m_s,
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
                self._motion = MinimumJerkPoseMotion.from_speed(
                    start_pos_w=ee_pos_w,
                    goal_pos_w=place_hand_pos,
                    quat_w=target_quat_w,
                    start_time_s=sim_time_s,
                    max_speed_m_s=self.spec.approach_max_speed_m_s,
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
                self._motion = MinimumJerkPoseMotion.from_speed(
                    start_pos_w=ee_pos_w,
                    goal_pos_w=place_transit_pos,
                    quat_w=target_quat_w,
                    start_time_s=sim_time_s,
                    max_speed_m_s=self.spec.retreat_max_speed_m_s,
                )
            pos, quat, finished = self._motion.sample(sim_time_s)
            target_pos_w = pos
            target_quat_w = quat
            if finished:
                self.state = "done"
                self._motion = None

        elif self.state == "done":
            target_pos_w = place_transit_pos
            finger_opening = self.spec.open_finger_m
            done = True

        return PolicyCommand(
            target_pos_w=target_pos_w,
            target_quat_w=target_quat_w,
            finger_opening_m=finger_opening,
            done=done,
        )
