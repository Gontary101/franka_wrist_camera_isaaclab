"""Reset logic for Franka tabletop episodes."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene

from franka_wrist_camera_scene.tasks.pick_place import PickPlaceTaskSpec
from franka_wrist_camera_scene.tasks.reaching import ReachingTaskSpec


def reset_robot_to_default(scene: InteractiveScene) -> None:
    """Reset the robot to its default root and joint state."""
    robot: Articulation = scene["robot"]
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(),
        robot.data.default_joint_vel.clone(),
    )
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())


def reset_pick_place_objects(scene: InteractiveScene, spec: PickPlaceTaskSpec) -> None:
    """Reset the pick-place object to the task initial pose and zero velocity."""
    obj = scene[spec.object_name]

    root_state = obj.data.default_root_state.clone()
    pos_local = torch.tensor(spec.object_pos_local, device=root_state.device).view(1, 3)

    root_state[:, :3] = scene.env_origins + pos_local
    root_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=root_state.device).view(1, 4)
    root_state[:, 7:] = 0.0

    obj.write_root_pose_to_sim(root_state[:, :7])
    obj.write_root_velocity_to_sim(root_state[:, 7:])


def reset_pick_place_episode(scene: InteractiveScene, spec: PickPlaceTaskSpec) -> None:
    """Reset robot and task objects for one deterministic pick-place episode."""
    reset_robot_to_default(scene)
    reset_pick_place_objects(scene, spec)
    scene.reset()


def reset_reaching_objects(scene: InteractiveScene, spec: ReachingTaskSpec) -> None:
    """Reset the reaching object to the task initial pose and zero velocity."""
    obj = scene[spec.object_name]

    root_state = obj.data.default_root_state.clone()
    pos_local = torch.tensor(spec.object_pos_local, device=root_state.device).view(1, 3)

    root_state[:, :3] = scene.env_origins + pos_local
    root_state[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=root_state.device).view(1, 4)
    root_state[:, 7:] = 0.0

    obj.write_root_pose_to_sim(root_state[:, :7])
    obj.write_root_velocity_to_sim(root_state[:, 7:])


def reset_reaching_episode(scene: InteractiveScene, spec: ReachingTaskSpec) -> None:
    """Reset robot and task objects for one deterministic reaching episode."""
    reset_robot_to_default(scene)
    reset_reaching_objects(scene, spec)
    scene.reset()

