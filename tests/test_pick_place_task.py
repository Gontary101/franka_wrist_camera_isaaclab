from math import isclose
from unittest import TestCase

from franka_wrist_camera_scene.tasks.pick_place import PickPlaceTaskSpec, make_pick_place_episode_spec


class PickPlaceTaskTest(TestCase):
    def test_make_pick_place_episode_spec_requires_bbox_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "bbox metadata"):
            make_pick_place_episode_spec(
                base_spec=PickPlaceTaskSpec(),
                object_xy_offset=(0.0, 0.0),
                place_xy_offset=(0.0, 0.0),
                object_label="object",
            )

    def test_make_pick_place_episode_spec_uses_bbox_height_for_object_and_place(self) -> None:
        spec = make_pick_place_episode_spec(
            base_spec=PickPlaceTaskSpec(),
            object_xy_offset=(0.01, -0.02),
            place_xy_offset=(-0.03, 0.04),
            object_label="box",
            object_local_bbox_min=(-0.02, -0.03, -0.024),
            object_local_bbox_max=(0.02, 0.03, 0.04),
        )

        self.assertEqual(spec.object_pos_local[:2], (0.59, -0.18))
        self.assertTrue(isclose(spec.object_pos_local[2], 1.08))
        self.assertEqual(spec.place_pos_local[:2], (0.52, 0.26))
        self.assertTrue(isclose(spec.place_pos_local[2], 1.08))
