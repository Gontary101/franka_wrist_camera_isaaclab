"""Isaac Lab scene configuration for a Franka tabletop setup with cameras."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

from .settings import ROBOT_BASE_POS, TABLE_HEIGHT_M, TABLE_SCALE

WAREHOUSE_USD = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
TABLE_USD = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"


def pinhole_camera_cfg(clipping_range: tuple[float, float]) -> sim_utils.PinholeCameraCfg:
    """Return a compact RGB-D pinhole camera model."""
    return sim_utils.PinholeCameraCfg(
        focal_length=18.0,
        focus_distance=0.55,
        horizontal_aperture=20.955,
        clipping_range=clipping_range,
    )


@configclass
class TabletopFrankaSceneCfg(InteractiveSceneCfg):
    """Warehouse tabletop scene with a Franka Panda and two camera sensors."""

    warehouse = AssetBaseCfg(
        prim_path="/World/Warehouse",
        spawn=sim_utils.UsdFileCfg(usd_path=WAREHOUSE_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-4.0, -2.0, 0.0)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(usd_path=TABLE_USD, scale=TABLE_SCALE),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, TABLE_HEIGHT_M)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2200.0, color=(0.78, 0.78, 0.78)),
    )

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.fix_base = True
    robot.init_state.pos = ROBOT_BASE_POS

    target_cube = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/TargetCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.075, 0.075, 0.075),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.15, 0.10)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.58, -0.16, TABLE_HEIGHT_M + 0.08)),
    )

    blue_block = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/BlueBlock",
        spawn=sim_utils.CuboidCfg(
            size=(0.10, 0.055, 0.055),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.20, 0.85)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.43, 0.18, TABLE_HEIGHT_M + 0.065)),
    )

    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_rgbd_camera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        update_latest_camera_pose=True,
        spawn=pinhole_camera_cfg(clipping_range=(0.02, 4.0)),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.042, 0.0, 0.020),
            rot=(0.7054, -0.0493, 0.0493, -0.7054),
            convention="ros",
        ),
    )

    agent_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/AgentViewCamera",
        update_period=1.0 / 30.0,
        height=720,
        width=1280,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=pinhole_camera_cfg(clipping_range=(0.05, 25.0)),
        offset=CameraCfg.OffsetCfg(
            pos=(2.2, -2.2, 1.9),
            rot=(0.42976647, -0.13176112, 0.06355309, 0.89101111),
            convention="world",
        ),
    )
