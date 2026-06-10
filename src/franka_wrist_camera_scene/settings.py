"""Shared scene and motion parameters for the Franka tabletop demo."""

from __future__ import annotations

TABLE_HEIGHT_M = 1.05
TABLE_SCALE = (1.25, 0.85, 1.0)

ROBOT_BASE_POS = (0.1, 0.0, TABLE_HEIGHT_M)

# Local to each Isaac Lab environment origin.
CIRCLE_CENTER_LOCAL = (0.45, 0.0, TABLE_HEIGHT_M + 0.26)
CIRCLE_DIAMETER_M = 0.40
CIRCLE_FREQUENCY_HZ = 0.045

# WXYZ quaternion used by Isaac Lab. This is the standard Franka tabletop IK orientation
# used to keep the Panda hand directed toward the table during the circle motion.
GRIPPER_DOWN_QUAT_WXYZ = (0.0, 1.0, 0.0, 0.0)
