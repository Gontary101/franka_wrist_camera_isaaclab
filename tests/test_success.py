import sys
from unittest.mock import MagicMock

# Mock Isaac Lab modules to run tests in pure Python environment
sys.modules['isaaclab'] = MagicMock()
sys.modules['isaaclab.scene'] = MagicMock()
sys.modules['isaaclab.utils'] = MagicMock()
sys.modules['isaaclab.utils.math'] = MagicMock()

from unittest import TestCase
import torch

from franka_wrist_camera_scene.episode.success import (
    receptacle_xy_radius_from_bbox,
    pick_place_success,
)
from franka_wrist_camera_scene.tasks.pick_place import PickPlaceTaskSpec


class TestSuccess(TestCase):
    def test_receptacle_xy_radius_from_bbox(self) -> None:
        bbox_min = (-0.1, -0.2, -0.05)
        bbox_max = (0.1, 0.2, 0.05)
        
        # size_x = 0.2, size_y = 0.4
        # min(size_x, size_y) = 0.2
        # radius = 0.5 * 0.2 + 0.025 = 0.125
        radius = receptacle_xy_radius_from_bbox(bbox_min, bbox_max, margin_m=0.025)
        self.assertAlmostEqual(radius, 0.125)

    def test_pick_place_success_receptacle(self) -> None:
        # Mock scene and object
        mock_obj = MagicMock()
        # Shape (1, 3): 1 env
        mock_obj.data.root_pos_w = torch.tensor([[0.5, 0.1, 1.1]])
        
        mock_scene = MagicMock()
        mock_scene.__getitem__.return_value = mock_obj
        mock_scene.env_origins = torch.tensor([[0.0, 0.0, 0.0]])
        
        spec = PickPlaceTaskSpec(
            object_name="box",
            placement_target_pos_local=(0.5, 0.1, 1.0),
            object_local_bbox_min=(-0.02, -0.02, -0.02),
            object_local_bbox_max=(0.02, 0.02, 0.02),
            placement_target_local_bbox_min=(-0.1, -0.1, -0.05),
            placement_target_local_bbox_max=(0.1, 0.1, 0.05),
        )
        
        # receptacle_xy_radius = 0.5 * 0.2 + 0.025 = 0.125
        # xy_error = 0.0 (object at 0.5, 0.1, receptacle center at 0.5, 0.1)
        # object_bottom_z = 1.1 - 0.02 = 1.08
        # receptacle_top_z = 1.0 + 0.05 = 1.05
        # receptacle_bottom_z = 1.0 - 0.05 = 0.95
        # object_bottom_z (1.08) >= 0.92 (0.95 - 0.03) and <= 1.10 (1.05 + 0.05) -> True
        success = pick_place_success(mock_scene, spec)
        self.assertTrue(success.item())

        # Test failure: object too high
        mock_obj.data.root_pos_w = torch.tensor([[0.5, 0.1, 1.15]]) # bottom z = 1.13 > 1.10
        success = pick_place_success(mock_scene, spec)
        self.assertFalse(success.item())

        # Test failure: object too low (falls through)
        mock_obj.data.root_pos_w = torch.tensor([[0.5, 0.1, 0.90]]) # bottom z = 0.88 < 0.92
        success = pick_place_success(mock_scene, spec)
        self.assertFalse(success.item())

        # Test failure: object too far horizontally
        mock_obj.data.root_pos_w = torch.tensor([[0.65, 0.1, 1.1]]) # xy error = 0.15 > 0.125
        success = pick_place_success(mock_scene, spec)
        self.assertFalse(success.item())
