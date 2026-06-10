"""Shared settings loading constants from configs/scene.yaml to prevent drift."""

from __future__ import annotations

from .utils.paths import load_yaml_config

# Load config from the single source of truth configs/scene.yaml
_cfg = load_yaml_config("scene.yaml")

TABLE_HEIGHT_M = _cfg["table"]["height_m"]
TABLE_SIZE = tuple(_cfg["table"]["size"])
ROBOT_BASE_POS = tuple(_cfg["robot"]["base_pos"])

# Local to each Isaac Lab environment origin.
CIRCLE_CENTER_LOCAL = tuple(_cfg["debug_circle"]["center_local"])
CIRCLE_DIAMETER_M = _cfg["debug_circle"]["diameter_m"]
CIRCLE_FREQUENCY_HZ = _cfg["debug_circle"]["frequency_hz"]

# WXYZ quaternion used by Isaac Lab to orient the gripper toward the table.
GRIPPER_DOWN_QUAT_WXYZ = tuple(_cfg["debug_circle"]["orientation_wxyz"])
