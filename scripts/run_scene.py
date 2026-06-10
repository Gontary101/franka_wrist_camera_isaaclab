#!/usr/bin/env python3
"""Run the Franka tabletop wrist-camera scene in Isaac Lab."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


# Inject compatibility layer for Isaac Sim 6.0 (redirects omni.physics.tensors.impl.api -> omni.physics.tensors.api)
class LazyApiModule(types.ModuleType):
    def __getattr__(self, name):
        import omni.physics.tensors.api as api

        if name == "SoftBodyView":
            return getattr(api, "DeformableBodyView")
        return getattr(api, name)

    def __dir__(self):
        import omni.physics.tensors.api as api

        return dir(api)


sys.modules["omni.physics.tensors.impl.api"] = LazyApiModule("omni.physics.tensors.impl.api")
sys.modules["omni.physics.tensors.impl"] = types.ModuleType("omni.physics.tensors.impl")

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from isaaclab.app import AppLauncher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Franka Panda tabletop scene with wrist and agent cameras.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of cloned tabletop scenes.")
    parser.add_argument(
        "--max_steps", type=int, default=0, help="Stop after this many simulation steps; 0 runs forever."
    )
    parser.add_argument(
        "--circle_diameter", type=float, default=0.40, help="Gripper circle diameter in meters."
    )
    parser.add_argument("--circle_frequency", type=float, default=0.045, help="Circle frequency in Hz.")
    parser.add_argument("--probe_u", type=int, default=320, help="Wrist-camera pixel u coordinate.")
    parser.add_argument("--probe_v", type=int, default=240, help="Wrist-camera pixel v coordinate.")
    parser.add_argument(
        "--save_probe_every", type=int, default=0, help="Save wrist-camera overlay every N steps; 0 disables."
    )
    parser.add_argument("--video", action="store_true", help="Record a video from the wrist camera.")
    parser.add_argument(
        "--video_steps", type=int, default=7200, help="Number of steps to record for the video."
    )
    parser.add_argument(
        "--show_markers", action="store_true", help="Show physical circle debug markers in the scene."
    )
    parser.add_argument(
        "--viewport_camera",
        choices=("agent", "wrist", "perspective"),
        default="agent",
        help="Camera shown in the active viewport after scene initialization.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.kit_args = f"{args.kit_args} --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false".strip()
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Patch pxr.PhysxSchema for Isaac Sim 6.0 compatibility
from pxr import Gf, PhysxSchema, UsdGeom

if not hasattr(PhysxSchema, "PhysxDeformableBodyAPI"):
    PhysxSchema.PhysxDeformableBodyAPI = PhysxSchema.PhysxRigidBodyAPI

# Patch CreateShaderPrimFromSdrCommand for Isaac Sim 6.0 compatibility (name -> prim_name)
import omni.usd.commands

original_init = omni.usd.commands.CreateShaderPrimFromSdrCommand.__init__


def patched_init(self, *args, **kwargs):
    if "name" in kwargs:
        kwargs["prim_name"] = kwargs.pop("name")
    original_init(self, *args, **kwargs)


omni.usd.commands.CreateShaderPrimFromSdrCommand.__init__ = patched_init

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from franka_wrist_camera_scene import (  # noqa: E402
    CircleTrajectoryCfg,
    FrankaCircleIKController,
    TabletopFrankaSceneCfg,
    VideoRecorder,
    WristCameraProbe,
)
from franka_wrist_camera_scene.settings import CIRCLE_CENTER_LOCAL, GRIPPER_DOWN_QUAT_WXYZ  # noqa: E402

ENV0_PRIM = "/World/envs/env_0"
VIEWPORT_CAMERA_PRIMS = {
    "agent": f"{ENV0_PRIM}/AgentViewCamera",
    "wrist": f"{ENV0_PRIM}/Robot/panda_hand/wrist_rgbd_camera",
}


def resolved_env_0_path(prim_path: str) -> str:
    """Return the concrete env_0 prim path for a scene config path."""
    return prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")


def warm_start_cameras(sim: sim_utils.SimulationContext, scene: InteractiveScene, frames: int = 24) -> None:
    """Render a few stable frames before motion starts so camera render products bind cleanly."""
    sim_dt = sim.get_physics_dt()

    for _ in range(frames):
        scene.write_data_to_sim()
        sim.render()
        scene.update(sim_dt)

        for camera_name in ("wrist_camera", "agent_camera"):
            camera = scene[camera_name]
            camera.update(sim_dt, force_recompute=True)
            _ = camera.data.output["rgb"]


def sync_viewport_camera(camera_name: str) -> None:
    """Bind the active viewport to a scene camera after the camera prim has rendered once."""
    if camera_name == "perspective":
        return

    from isaacsim.core.rendering_manager import ViewportManager

    ViewportManager.set_camera(VIEWPORT_CAMERA_PRIMS[camera_name])
    sim_utils.SimulationContext.instance().render()
    ViewportManager.wait_for_viewport(max_frames=30)


def nudge_camera_prims(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    """Dirty camera transforms once, then restore them exactly."""
    stage = omni.usd.get_context().get_stage()

    for env_id in range(scene.num_envs):
        for env0_path in VIEWPORT_CAMERA_PRIMS.values():
            path = env0_path.replace("/env_0/", f"/env_{env_id}/")
            prim = stage.GetPrimAtPath(path)
            xform = UsdGeom.Xformable(prim)

            translate_op = next(
                (op for op in xform.GetOrderedXformOps() if op.GetOpName() == "xformOp:translate"),
                None,
            )
            if translate_op is None:
                translate_op = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)

            original = translate_op.Get() or Gf.Vec3d(0.0, 0.0, 0.0)
            translate_op.Set(Gf.Vec3d(original[0] + 1.0e-3, original[1], original[2]))
            sim.render()
            translate_op.Set(original)

    sim.render()


def reset_scene(scene: InteractiveScene) -> None:
    """Write the default robot state once before simulation starts."""
    robot: Articulation = scene["robot"]
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone())
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    scene.reset()


def build_controller() -> FrankaCircleIKController:
    trajectory = CircleTrajectoryCfg(
        center_local=CIRCLE_CENTER_LOCAL,
        diameter_m=args_cli.circle_diameter,
        frequency_hz=args_cli.circle_frequency,
        orientation_wxyz=GRIPPER_DOWN_QUAT_WXYZ,
    )
    return FrankaCircleIKController(trajectory=trajectory, show_markers=args_cli.show_markers)


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    controller: FrankaCircleIKController,
    probe: WristCameraProbe,
    max_steps: int,
    video: bool = False,
    video_steps: int = 0,
) -> None:
    """Run the scene until the app closes or the optional step limit is reached."""
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()
    sim_time_s = 0.0
    step = 0

    video_recorder = VideoRecorder(video, sim_dt)

    while simulation_app.is_running() and (max_steps <= 0 or step < max_steps):
        controller.apply(scene, robot, sim_time_s)
        scene.write_data_to_sim()

        sim.step()
        sim_time_s += sim_dt
        step += 1
        scene.update(sim_dt)
        probe.maybe_save(scene, step)

        video_recorder.record_step(scene, step)

    video_recorder.close()


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        device=args_cli.device,
        physx=sim_utils.PhysxCfg(
            enable_external_forces_every_iteration=True,
            min_velocity_iteration_count=1,
            min_position_iteration_count=4,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.2, -2.2, 1.9], target=[0.55, 0.0, 1.20])

    scene = InteractiveScene(TabletopFrankaSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5))
    controller = build_controller()
    probe = WristCameraProbe(args_cli.probe_u, args_cli.probe_v, args_cli.save_probe_every)

    sim.reset()
    controller.bind(scene, scene["robot"])
    reset_scene(scene)
    controller.reset()

    warm_start_cameras(sim, scene)

    if not args_cli.headless:
        sync_viewport_camera(args_cli.viewport_camera)

    nudge_camera_prims(sim, scene)
    run_simulator(sim, scene, controller, probe, args_cli.max_steps, args_cli.video, args_cli.video_steps)


if __name__ == "__main__":
    main()
    simulation_app.close()
