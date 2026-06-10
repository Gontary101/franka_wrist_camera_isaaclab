#!/usr/bin/env python3
"""Run the Franka tabletop wrist-camera scene in Isaac Lab."""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path


# Compatibility layer for Isaac Sim 6.0 (redirects omni.physics.tensors.impl.api -> omni.physics.tensors.api)
class LazyApiModule(types.ModuleType):
    def __getattr__(self, name):
        import omni.physics.tensors.api as api
        return getattr(api, "DeformableBodyView" if name == "SoftBodyView" else name)

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
        "--show_markers", action="store_true", help="Show physical circle debug markers in the scene."
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
from pxr import PhysxSchema  # noqa: E402

if not hasattr(PhysxSchema, "PhysxDeformableBodyAPI"):
    PhysxSchema.PhysxDeformableBodyAPI = PhysxSchema.PhysxRigidBodyAPI

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from franka_wrist_camera_scene import (  # noqa: E402
    CircleTrajectoryCfg,
    CartesianIKController,
    GripperController,
    CircleMotionPolicy,
    TabletopFrankaSceneCfg,
    VideoRecorder,
    WristCameraProbe,
    circle_points_w,
)
from franka_wrist_camera_scene.debug.visualization import CircleMotionMarkers  # noqa: E402
from franka_wrist_camera_scene.settings import CIRCLE_CENTER_LOCAL, GRIPPER_DOWN_QUAT_WXYZ  # noqa: E402


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


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    policy: CircleMotionPolicy,
    ik: CartesianIKController,
    gripper: GripperController,
    probe: WristCameraProbe,
    max_steps: int,
    video: bool = False,
    show_markers: bool = False,
) -> None:
    """Run the scene until the app closes or the optional step limit is reached."""
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()
    sim_time_s = 0.0
    step = 0

    video_recorder = VideoRecorder(video, sim_dt)

    # Debug markers
    markers = None
    if show_markers:
        markers = CircleMotionMarkers()
        points_w = circle_points_w(scene, policy.cfg, robot.device)
        markers.draw_path(points_w)

    while simulation_app.is_running() and (max_steps <= 0 or step < max_steps):
        # 1. Step the policy to get reference actions
        target_pos_w, target_quat_w, gripper_width = policy.step(None, sim_time_s)

        # 2. Update and apply Cartesian IK command
        ik.set_target_pose(target_pos_w, target_quat_w)
        ik.apply(scene, robot)

        # 3. Update and apply gripper command
        gripper.set_width(gripper_width)
        gripper.apply(robot)

        scene.write_data_to_sim()

        sim.step()
        sim_time_s += sim_dt
        step += 1
        scene.update(sim_dt)
        probe.maybe_save(scene, step)

        if markers is not None:
            markers.draw_target(target_pos_w)

        video_recorder.record_step(scene, step)

    video_recorder.close()


def nudge_camera_prims(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    """Dirty camera transforms once to prevent white camera views."""
    from pxr import Gf, UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    for camera_name in ("wrist_camera", "agent_camera"):
        camera = scene[camera_name]
        for path in camera._view.prim_paths:
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
    robot: Articulation = scene["robot"]

    trajectory_cfg = CircleTrajectoryCfg(
        center_local=CIRCLE_CENTER_LOCAL,
        diameter_m=args_cli.circle_diameter,
        frequency_hz=args_cli.circle_frequency,
        orientation_wxyz=GRIPPER_DOWN_QUAT_WXYZ,
    )

    policy = CircleMotionPolicy(cfg=trajectory_cfg)
    ik = CartesianIKController()
    gripper = GripperController()
    probe = WristCameraProbe(args_cli.probe_u, args_cli.probe_v, args_cli.save_probe_every)

    sim.reset()
    policy.bind(scene, robot)
    ik.bind(scene, robot)
    gripper.bind(scene, robot)
    reset_scene(scene)
    ik.reset()

    nudge_camera_prims(sim, scene)
    run_simulator(
        sim,
        scene,
        policy,
        ik,
        gripper,
        probe,
        args_cli.max_steps,
        video=args_cli.video,
        show_markers=args_cli.show_markers,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
