"""Success predicates for tabletop episodes."""

from __future__ import annotations

import torch
from isaaclab.scene import InteractiveScene

from franka_wrist_camera_scene.tasks.pick_place import PickPlaceTaskSpec
from franka_wrist_camera_scene.tasks.reaching import ReachingTaskSpec


def pick_place_success(
    scene: InteractiveScene,
    spec: PickPlaceTaskSpec,
    xy_threshold_m: float = 0.08,
    z_threshold_m: float = 0.08,
) -> torch.Tensor:
    """Return per-env success for placing the object near the target area."""
    obj = scene[spec.object_name]
    obj_pos_w = obj.data.root_pos_w

    target_pos_local = torch.tensor(spec.place_pos_local, device=obj_pos_w.device).view(1, 3)
    target_pos_w = scene.env_origins + target_pos_local

    xy_error = torch.linalg.norm(obj_pos_w[:, :2] - target_pos_w[:, :2], dim=-1)
    z_error = torch.abs(obj_pos_w[:, 2] - target_pos_w[:, 2])

    return (xy_error <= xy_threshold_m) & (z_error <= z_threshold_m)


def reaching_success(
    scene: InteractiveScene,
    spec: ReachingTaskSpec,
    threshold_m: float = 0.05,
) -> torch.Tensor:
    """Return per-env success if the robot end-effector is close to the target object."""
    robot = scene["robot"]
    ee_body_id = robot.find_bodies(spec.ee_body_name)[0][0]
    ee_pos_w = robot.data.body_pos_w[:, ee_body_id]

    obj = scene[spec.object_name]
    obj_pos_w = obj.data.root_pos_w[:, :3]

    distance = torch.linalg.norm(ee_pos_w - obj_pos_w, dim=-1)
    return distance <= threshold_m

