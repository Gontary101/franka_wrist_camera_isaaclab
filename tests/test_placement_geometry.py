from math import isclose

from franka_wrist_camera_scene.tasks.placement_geometry import (
    object_root_pose_on_support,
    object_root_z_on_support,
)


def test_object_root_z_on_support_places_bbox_bottom_above_surface() -> None:
    root_z = object_root_z_on_support(
        support_surface_z=1.05,
        object_bbox_min_z=-0.024,
        bottom_clearance_m=0.006,
    )

    assert isclose(root_z, 1.08)


def test_object_root_pose_on_support_keeps_xy_and_derives_z() -> None:
    root_pose = object_root_pose_on_support(
        xy_pos=(0.58, -0.16),
        support_surface_z=1.05,
        object_bbox_min_z=-0.024,
        bottom_clearance_m=0.006,
    )

    assert root_pose[:2] == (0.58, -0.16)
    assert isclose(root_pose[2], 1.08)
