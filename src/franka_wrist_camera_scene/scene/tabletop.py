"""Isaac Lab scene configuration for a Franka tabletop setup with cameras."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG

from ..settings import ROBOT_BASE_POS, TABLE_HEIGHT_M, TABLE_SIZE
from franka_wrist_camera_scene.scene.object_context import CatalogObjectContext

WAREHOUSE_USD = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"


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
        spawn=sim_utils.CuboidCfg(
            size=TABLE_SIZE,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.42, 0.30)),
        ),
        # Cuboid origin is at its center; keep TABLE_HEIGHT_M as the tabletop z.
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.45, 0.0, TABLE_HEIGHT_M - 0.5 * TABLE_SIZE[2])),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=900.0, color=(0.9, 0.9, 0.9)),
    )

    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.fix_base = True
    robot.init_state.pos = ROBOT_BASE_POS

    target_cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetCube",
        spawn=sim_utils.UsdFileCfg(
            usd_path="",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.58, -0.16, TABLE_HEIGHT_M + 0.05)),
    )

    place_target = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PlaceTarget",
        spawn=sim_utils.CuboidCfg(
            size=(0.14, 0.14, 0.004),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.65, 0.20)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.22, TABLE_HEIGHT_M + 0.002)),
    )

    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_rgbd_camera",
        update_period=0.0,
        height=128,
        width=128,
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
        height=128,
        width=128,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=pinhole_camera_cfg(clipping_range=(0.05, 25.0)),
        offset=CameraCfg.OffsetCfg(
            pos=(1.4186131747, 0.0, 1.7603500240),
            rot=(0.0, -0.33316794, 0.0, 0.94286750),
            convention="world",
        ),
    )


def make_tabletop_scene_cfg(
    object_context: CatalogObjectContext,
    num_envs: int = 1,
    env_spacing: float = 2.5,
) -> TabletopFrankaSceneCfg:
    """Create a tabletop scene configuration with the specified target object."""
    scene_cfg = TabletopFrankaSceneCfg(num_envs=num_envs, env_spacing=env_spacing)
    scene_cfg.target_cube.spawn.usd_path = str(object_context.usd_path)
    return scene_cfg
