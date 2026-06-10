"""Task-space control nodes for the Franka tabletop scene."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from ..settings import CIRCLE_CENTER_LOCAL, CIRCLE_DIAMETER_M, CIRCLE_FREQUENCY_HZ, GRIPPER_DOWN_QUAT_WXYZ
from ..debug.visualization import CircleMotionMarkers


@dataclass(frozen=True, slots=True)
class CircleTrajectoryCfg:
    """Circular end-effector path expressed in each environment's local frame."""

    center_local: tuple[float, float, float] = CIRCLE_CENTER_LOCAL
    diameter_m: float = CIRCLE_DIAMETER_M
    frequency_hz: float = CIRCLE_FREQUENCY_HZ
    orientation_wxyz: tuple[float, float, float, float] = GRIPPER_DOWN_QUAT_WXYZ
    tcp_offset_local: tuple[float, float, float] = (0.0, 0.0, 0.10)
    preview_points: int = 96

    @property
    def radius_m(self) -> float:
        return 0.5 * self.diameter_m


@dataclass(slots=True)
class FrankaCircleIKController:
    """Drive the Panda gripper around a horizontal circle using differential IK."""

    trajectory: CircleTrajectoryCfg = field(default_factory=CircleTrajectoryCfg)
    open_finger_width_m: float = 0.035
    arm_joint_expr: str = "panda_joint.*"
    finger_joint_expr: str = "panda_finger_joint.*"
    end_effector_body: str = "panda_hand"
    show_markers: bool = False

    _entity: SceneEntityCfg = field(init=False, repr=False)
    _ik: DifferentialIKController = field(init=False, repr=False)
    _ee_jacobian_index: int = field(init=False, repr=False)
    _finger_joint_ids: list[int] = field(init=False, repr=False)
    _finger_target: torch.Tensor = field(init=False, repr=False)
    _center_local: torch.Tensor = field(init=False, repr=False)
    _target_quat_w: torch.Tensor = field(init=False, repr=False)
    _tcp_offset_local: torch.Tensor = field(init=False, repr=False)
    _circle_points_local: torch.Tensor = field(init=False, repr=False)
    _markers: CircleMotionMarkers | None = field(init=False, default=None, repr=False)

    def bind(self, scene: InteractiveScene, robot: Articulation) -> None:
        """Resolve scene ids, allocate controller buffers, and draw the desired circle."""
        self._entity = SceneEntityCfg(
            "robot",
            joint_names=[self.arm_joint_expr],
            body_names=[self.end_effector_body],
        )
        self._entity.resolve(scene)

        self._ee_jacobian_index = self._entity.body_ids[0] - int(robot.is_fixed_base)
        self._finger_joint_ids, _ = robot.find_joints(self.finger_joint_expr)
        self._finger_target = torch.full(
            (scene.num_envs, len(self._finger_joint_ids)),
            self.open_finger_width_m,
            device=robot.device,
        )

        self._ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose",
                use_relative_mode=False,
                ik_method="dls",
                ik_params={"lambda_val": 0.01},
            ),
            num_envs=scene.num_envs,
            device=robot.device,
        )

        self._center_local = torch.tensor(self.trajectory.center_local, device=robot.device).view(1, 3)
        self._target_quat_w = torch.tensor(self.trajectory.orientation_wxyz, device=robot.device).view(1, 4)
        self._tcp_offset_local = torch.tensor(self.trajectory.tcp_offset_local, device=robot.device).view(1, 3)
        self._circle_points_local = self._build_circle_points(robot.device)
        
        if self.show_markers:
            self._markers = CircleMotionMarkers()
            self._markers.draw_path(self.circle_points_w(scene))

    def reset(self) -> None:
        """Reset the differential IK solver state."""
        self._ik.reset()

    def apply(self, scene: InteractiveScene, robot: Articulation, sim_time_s: float) -> None:
        """Compute and write arm and gripper joint targets for the current simulation time."""
        target_pos_w, target_quat_w = self.target_pose_w(scene, sim_time_s)
        target_pos_b, target_quat_b = self._target_pose_in_base(robot, target_pos_w, target_quat_w)
        self._ik.set_command(torch.cat((target_pos_b, target_quat_b), dim=-1))

        joint_pos_des = self._compute_joint_target(robot)
        robot.set_joint_position_target(joint_pos_des, joint_ids=self._entity.joint_ids)
        robot.set_joint_position_target(self._finger_target, joint_ids=self._finger_joint_ids)
        
        if self._markers is not None:
            self._markers.draw_target(target_pos_w)

    def target_pose_w(self, scene: InteractiveScene, sim_time_s: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the desired panda_hand pose that places the TCP on the circle."""
        phase = 2.0 * math.pi * self.trajectory.frequency_hz * sim_time_s

        tcp_pos_w = self._center_local.repeat(scene.num_envs, 1) + scene.env_origins
        tcp_pos_w[:, 0] += self.trajectory.radius_m * math.cos(phase)
        tcp_pos_w[:, 1] += self.trajectory.radius_m * math.sin(phase)

        target_quat_w = self._target_quat_w.repeat(scene.num_envs, 1)

        tcp_offset_w = quat_apply(target_quat_w, self._tcp_offset_local.repeat(scene.num_envs, 1))
        hand_pos_w = tcp_pos_w - tcp_offset_w

        return hand_pos_w, target_quat_w

    def circle_points_w(self, scene: InteractiveScene) -> torch.Tensor:
        """Return desired circle preview points in world coordinates for every environment."""
        points = self._circle_points_local.unsqueeze(0) + scene.env_origins.unsqueeze(1)
        return points.reshape(-1, 3)

    def _build_circle_points(self, device: str) -> torch.Tensor:
        angles = torch.linspace(0.0, 2.0 * math.pi, self.trajectory.preview_points + 1, device=device)[:-1]
        points = self._center_local.repeat(self.trajectory.preview_points, 1)
        points[:, 0] += self.trajectory.radius_m * torch.cos(angles)
        points[:, 1] += self.trajectory.radius_m * torch.sin(angles)
        return points

    def _target_pose_in_base(
        self,
        robot: Articulation,
        target_pos_w: torch.Tensor,
        target_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        root_pose_w = robot.data.root_pose_w
        return subtract_frame_transforms(root_pose_w[:, :3], root_pose_w[:, 3:7], target_pos_w, target_quat_w)

    def _compute_joint_target(self, robot: Articulation) -> torch.Tensor:
        jacobian = robot.root_physx_view.get_jacobians()[:, self._ee_jacobian_index, :, self._entity.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, self._entity.body_ids[0]]
        root_pose_w = robot.data.root_pose_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, :3],
            ee_pose_w[:, 3:7],
        )
        joint_pos = robot.data.joint_pos[:, self._entity.joint_ids]
        return self._ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
