"""Reset logic for Franka tabletop episodes."""

from __future__ import annotations

from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene


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
    scene.reset()
