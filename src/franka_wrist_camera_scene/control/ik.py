"""Inverse Kinematics solver using Isaac Lab differential IK."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import subtract_frame_transforms


class CartesianIKController:
    """Robot arm end-effector IK controller using differential IK."""

    def __init__(
        self,
        arm_joint_expr: str = "panda_joint.*",
        end_effector_body: str = "panda_hand",
    ):
        self.arm_joint_expr = arm_joint_expr
        self.end_effector_body = end_effector_body

        self._entity = None
        self._ik = None
        self._ee_jacobian_index = None
        self._target_pos_w = None
        self._target_quat_w = None

    def bind(self, scene: InteractiveScene, robot: Articulation) -> None:
        """Resolve scene references and initialize differential IK."""
        self._entity = SceneEntityCfg(
            "robot",
            joint_names=[self.arm_joint_expr],
            body_names=[self.end_effector_body],
        )
        self._entity.resolve(scene)

        self._ee_jacobian_index = self._entity.body_ids[0] - int(robot.is_fixed_base)

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

    def reset(self) -> None:
        """Reset the differential IK solver state."""
        self._ik.reset()

    def set_target_pose(self, target_pos_w: torch.Tensor, target_quat_w: torch.Tensor) -> None:
        """Set the target end-effector pose in world coordinates."""
        self._target_pos_w = target_pos_w
        self._target_quat_w = target_quat_w

    def apply(self, scene: InteractiveScene, robot: Articulation) -> None:
        """Compute and apply joint command targets for the arm."""
        if self._target_pos_w is None or self._target_quat_w is None:
            raise RuntimeError("CartesianIKController target pose was not set before apply().")

        # Transform target pose from world to robot base frame
        root_pose_w = robot.data.root_pose_w
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            self._target_pos_w,
            self._target_quat_w,
        )

        self._ik.set_command(torch.cat((target_pos_b, target_quat_b), dim=-1))

        # Compute joint velocities/positions from Jacobian and current joint states
        jacobian = robot.root_physx_view.get_jacobians()[:, self._ee_jacobian_index, :, self._entity.joint_ids]
        ee_pose_w = robot.data.body_pose_w[:, self._entity.body_ids[0]]
        
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, :3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, :3],
            ee_pose_w[:, 3:7],
        )
        joint_pos = robot.data.joint_pos[:, self._entity.joint_ids]
        joint_pos_des = self._ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        robot.set_joint_position_target(joint_pos_des, joint_ids=self._entity.joint_ids)
