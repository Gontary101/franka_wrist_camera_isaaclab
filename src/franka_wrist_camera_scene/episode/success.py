"""Success predicates for tabletop episodes."""

from __future__ import annotations

import torch
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import quat_apply

from franka_wrist_camera_scene.tasks.pick_place import PickPlaceTaskSpec
from franka_wrist_camera_scene.tasks.reaching import ReachingTaskSpec


def receptacle_xy_radius_from_bbox(
    bbox_min: tuple[float, float, float],
    bbox_max: tuple[float, float, float],
    margin_m: float,
) -> float:
    size_x = float(bbox_max[0]) - float(bbox_min[0])
    size_y = float(bbox_max[1]) - float(bbox_min[1])
    return 0.5 * min(size_x, size_y) + margin_m


def pick_place_success(
    scene: InteractiveScene,
    spec: PickPlaceTaskSpec,
    xy_threshold_m: float = 0.08,
    z_threshold_m: float = 0.08,
) -> torch.Tensor:
    """Return per-env success for placing the object near the target area."""
    obj = scene[spec.object_name]
    obj_pos_w = obj.data.root_pos_w

    if spec.placement_target_pos_local is not None:
        if (
            spec.object_local_bbox_min is None
            or spec.placement_target_local_bbox_min is None
            or spec.placement_target_local_bbox_max is None
        ):
            raise RuntimeError("Receptacle placement success requires object and placement target geometry.")

        receptacle_pos_local = torch.tensor(spec.placement_target_pos_local, device=obj_pos_w.device).view(1, 3)
        receptacle_pos_w = scene.env_origins + receptacle_pos_local
        xy_error = torch.linalg.norm(obj_pos_w[:, :2] - receptacle_pos_w[:, :2], dim=-1)

        xy_threshold = receptacle_xy_radius_from_bbox(
            bbox_min=spec.placement_target_local_bbox_min,
            bbox_max=spec.placement_target_local_bbox_max,
            margin_m=0.025,
        )

        object_bottom_z = obj_pos_w[:, 2] + float(spec.object_local_bbox_min[2])
        receptacle_top_z = receptacle_pos_w[:, 2] + float(spec.placement_target_local_bbox_max[2])
        receptacle_bottom_z = receptacle_pos_w[:, 2] + float(spec.placement_target_local_bbox_min[2])
        vertical_ok = (
            (object_bottom_z >= receptacle_bottom_z - 0.03)
            & (object_bottom_z <= receptacle_top_z + 0.05)
        )
        return (xy_error <= xy_threshold) & vertical_ok

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
    """Return per-env success if the robot TCP is close to the target object."""
    robot = scene["robot"]
    ee_body_id = robot.find_bodies(spec.ee_body_name)[0][0]
    ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
    ee_pos_w = ee_pose_w[:, :3]
    ee_quat_w = ee_pose_w[:, 3:7]

    tcp_offset_local = torch.tensor(spec.tcp_offset_local, device=ee_pos_w.device).view(1, 3)
    tcp_offset_w = quat_apply(ee_quat_w, tcp_offset_local.expand(ee_pos_w.shape[0], -1))
    tcp_pos_w = ee_pos_w + tcp_offset_w

    obj = scene[spec.object_name]
    obj_pos_w = obj.data.root_pos_w[:, :3]

    distance = torch.linalg.norm(tcp_pos_w - obj_pos_w, dim=-1)
    return distance <= threshold_m
