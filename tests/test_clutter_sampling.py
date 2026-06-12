"""Unit tests for geometry-aware clutter sampling."""

from __future__ import annotations

import math
import random
import unittest

from franka_wrist_camera_scene.scene.clutter import (
    FootprintDisk,
    XYRange,
    footprints_overlap,
    planar_footprint_radius_m,
    sample_non_overlapping_xy,
)


class ClutterSamplingTests(unittest.TestCase):
    def test_planar_footprint_radius_m_uses_bbox_diagonal_and_margin(self) -> None:
        radius = planar_footprint_radius_m(
            bbox_min=(-0.1, -0.2, 0.0),
            bbox_max=(0.1, 0.2, 0.1),
            margin_m=0.01,
        )
        expected = 0.5 * math.hypot(0.2, 0.4) + 0.01
        self.assertAlmostEqual(radius, expected)

    def test_footprints_overlap_detects_overlap_and_separation(self) -> None:
        overlapping_a = FootprintDisk(xy=(0.0, 0.0), radius_m=0.2)
        overlapping_b = FootprintDisk(xy=(0.3, 0.0), radius_m=0.2)
        separated_a = FootprintDisk(xy=(0.0, 0.0), radius_m=0.1)
        separated_b = FootprintDisk(xy=(1.0, 0.0), radius_m=0.1)

        self.assertTrue(footprints_overlap(overlapping_a, overlapping_b))
        self.assertFalse(footprints_overlap(separated_a, separated_b))

    def test_sample_non_overlapping_xy_avoids_occupied_disks(self) -> None:
        rng = random.Random(7)
        xy_range = XYRange(x=(0.0, 1.0), y=(0.0, 1.0))
        occupied = (FootprintDisk(xy=(0.2, 0.2), radius_m=0.15),)
        candidate_radius_m = 0.1

        for _ in range(32):
            xy = sample_non_overlapping_xy(
                rng=rng,
                xy_range=xy_range,
                candidate_radius_m=candidate_radius_m,
                occupied=occupied,
                max_attempts=64,
            )
            candidate = FootprintDisk(xy=xy, radius_m=candidate_radius_m)
            self.assertFalse(footprints_overlap(candidate, occupied[0]))

    def test_sample_non_overlapping_xy_fails_loudly_when_impossible(self) -> None:
        xy_range = XYRange(x=(0.0, 0.1), y=(0.0, 0.1))
        occupied = (FootprintDisk(xy=(0.05, 0.05), radius_m=1.0),)

        with self.assertRaises(RuntimeError):
            sample_non_overlapping_xy(
                rng=random.Random(0),
                xy_range=xy_range,
                candidate_radius_m=0.2,
                occupied=occupied,
                max_attempts=8,
            )


if __name__ == "__main__":
    unittest.main()
