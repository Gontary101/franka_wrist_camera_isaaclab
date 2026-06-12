from unittest import TestCase

import torch

from franka_wrist_camera_scene.control.motion_primitives import (
    MinimumJerkScalarProfile,
    MinimumJerkPoseMotion,
    MinimumJerkWaypointMotion,
)


class TestMotionPrimitives(TestCase):
    def test_minimum_jerk_scalar_profile(self) -> None:
        with self.assertRaises(ValueError):
            MinimumJerkScalarProfile(duration_s=0.0)
        with self.assertRaises(ValueError):
            MinimumJerkScalarProfile(duration_s=-1.0)

        profile = MinimumJerkScalarProfile(duration_s=2.0)

        alpha, done = profile.sample(-1.0)
        self.assertEqual(alpha, 0.0)
        self.assertFalse(done)

        alpha, done = profile.sample(0.0)
        self.assertEqual(alpha, 0.0)
        self.assertFalse(done)

        alpha, done = profile.sample(1.0)
        self.assertAlmostEqual(alpha, 0.5)
        self.assertFalse(done)

        alpha, done = profile.sample(2.0)
        self.assertAlmostEqual(alpha, 1.0)
        self.assertTrue(done)

        alpha, done = profile.sample(3.0)
        self.assertEqual(alpha, 1.0)
        self.assertTrue(done)

    def test_minimum_jerk_pose_motion(self) -> None:
        start_pos = torch.tensor([0.0, 0.0, 0.0])
        goal_pos = torch.tensor([1.0, 2.0, 3.0])
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        with self.assertRaises(ValueError):
            MinimumJerkPoseMotion.from_speed(
                start_pos_w=start_pos,
                goal_pos_w=goal_pos,
                quat_w=quat,
                start_time_s=1.0,
                max_speed_m_s=0.0,
            )

        motion = MinimumJerkPoseMotion.from_speed(
            start_pos_w=start_pos,
            goal_pos_w=goal_pos,
            quat_w=quat,
            start_time_s=1.0,
            max_speed_m_s=2.0,
        )

        self.assertGreater(motion.profile.duration_s, 0.0)

        pos, q, done = motion.sample(1.0)
        self.assertTrue(torch.allclose(pos, start_pos))
        self.assertTrue(torch.allclose(q, quat))
        self.assertFalse(done)

        end_time = 1.0 + motion.profile.duration_s
        pos, q, done = motion.sample(end_time)
        self.assertTrue(torch.allclose(pos, goal_pos))
        self.assertTrue(torch.allclose(q, quat))
        self.assertTrue(done)

    def test_minimum_jerk_pose_motion_respects_peak_speed_bound(self) -> None:
        start_pos = torch.tensor([0.0, 0.0, 0.0])
        goal_pos = torch.tensor([1.0, 0.0, 0.0])
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        motion = MinimumJerkPoseMotion.from_speed(
            start_pos_w=start_pos,
            goal_pos_w=goal_pos,
            quat_w=quat,
            start_time_s=0.0,
            max_speed_m_s=0.5,
        )

        self.assertAlmostEqual(motion.profile.duration_s, 3.75)

    def test_minimum_jerk_waypoint_motion_validates_inputs(self) -> None:
        point = torch.tensor([0.0, 0.0, 0.0])
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        with self.assertRaises(ValueError):
            MinimumJerkWaypointMotion.from_speed(
                waypoints_w=(point,),
                quat_w=quat,
                start_time_s=0.0,
                max_speed_m_s=1.0,
            )
        with self.assertRaises(ValueError):
            MinimumJerkWaypointMotion.from_segment_speeds(
                waypoints_w=(point, point.clone(), point.clone()),
                quat_w=quat,
                start_time_s=0.0,
                max_speed_m_s=(1.0,),
            )
        with self.assertRaises(ValueError):
            MinimumJerkWaypointMotion.from_speed(
                waypoints_w=(point, point.clone()),
                quat_w=quat,
                start_time_s=0.0,
                max_speed_m_s=0.0,
            )

    def test_minimum_jerk_waypoint_motion_samples_through_waypoints(self) -> None:
        start_pos = torch.tensor([0.0, 0.0, 0.0])
        middle_pos = torch.tensor([1.0, 0.0, 0.0])
        goal_pos = torch.tensor([2.0, 0.0, 0.0])
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        motion = MinimumJerkWaypointMotion.from_speed(
            waypoints_w=(start_pos, middle_pos, goal_pos),
            quat_w=quat,
            start_time_s=1.0,
            max_speed_m_s=0.5,
        )

        self.assertAlmostEqual(motion.profile.duration_s, 7.5)

        pos, q, done = motion.sample(1.0)
        self.assertTrue(torch.allclose(pos, start_pos))
        self.assertTrue(torch.allclose(q, quat))
        self.assertFalse(done)

        midpoint_time_s = 1.0 + 0.5 * motion.profile.duration_s
        pos, _, done = motion.sample(midpoint_time_s)
        self.assertTrue(torch.allclose(pos, middle_pos))
        self.assertFalse(done)

        pos, q, done = motion.sample(1.0 + motion.profile.duration_s)
        self.assertTrue(torch.allclose(pos, goal_pos))
        self.assertTrue(torch.allclose(q, quat))
        self.assertTrue(done)

    def test_minimum_jerk_waypoint_motion_respects_segment_speed_limits(self) -> None:
        start_pos = torch.tensor([0.0, 0.0, 0.0])
        middle_pos = torch.tensor([1.0, 0.0, 0.0])
        goal_pos = torch.tensor([2.0, 0.0, 0.0])
        quat = torch.tensor([1.0, 0.0, 0.0, 0.0])

        motion = MinimumJerkWaypointMotion.from_segment_speeds(
            waypoints_w=(start_pos, middle_pos, goal_pos),
            quat_w=quat,
            start_time_s=0.0,
            max_speed_m_s=(0.5, 1.0),
        )

        self.assertAlmostEqual(motion.profile.duration_s, 5.625)

        pos, _, _ = motion.sample(0.5 * motion.profile.duration_s)
        self.assertTrue(torch.allclose(pos, torch.tensor([0.75, 0.0, 0.0])))
