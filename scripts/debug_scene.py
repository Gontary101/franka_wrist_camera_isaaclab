#!/usr/bin/env python3
"""Run the Franka tabletop wrist-camera scene in Isaac Lab."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from franka_wrist_camera_scene.app import launcher  # noqa: F401
from isaaclab.app import AppLauncher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Franka Panda tabletop scene with wrist and agent cameras.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of cloned tabletop scenes.")
    parser.add_argument(
        "--max_steps", type=int, default=0, help="Stop after this many simulation steps; 0 runs forever."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="circle",
        choices=["circle", "pick_place", "reaching"],
        help="Task/policy to run.",
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
    parser.add_argument(
        "--collection_config",
        type=str,
        default="collection.yaml",
        help="Collection config used to resolve the target catalog object.",
    )
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.kit_args = f"{args.kit_args} --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false".strip()
    return args


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

launcher.patch_physx_schema()

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.scene import InteractiveScene  # noqa: E402

from franka_wrist_camera_scene.control.gripper import GripperController
from franka_wrist_camera_scene.control.ik import CartesianIKController
from franka_wrist_camera_scene.control.trajectory import CircleTrajectoryCfg, circle_points_w
from franka_wrist_camera_scene.debug.camera_probe import WristCameraProbe
from franka_wrist_camera_scene.debug.video_recorder import VideoRecorder
from franka_wrist_camera_scene.debug.visualization import CircleMotionMarkers
from franka_wrist_camera_scene.policies.circle_policy import CircleMotionPolicy
from franka_wrist_camera_scene.policies import PickPlaceScriptedPolicy, ReachingScriptedPolicy
from franka_wrist_camera_scene.scene.tabletop import make_tabletop_scene_cfg
from franka_wrist_camera_scene.scene.object_context import load_catalog_object_context
from franka_wrist_camera_scene.settings import CIRCLE_CENTER_LOCAL, GRIPPER_DOWN_QUAT_WXYZ, SIM_DT
from franka_wrist_camera_scene.tasks import (
    PickPlaceTaskSpec,
    make_pick_place_episode_spec,
    ReachingTaskSpec,
    make_reaching_episode_spec,
)
from franka_wrist_camera_scene.utils.paths import load_yaml_config
from franka_wrist_camera_scene.app.camera_warmup import nudge_camera_prims
from franka_wrist_camera_scene.episode.reset import (
    reset_robot_to_default,
    reset_pick_place_episode,
    reset_reaching_episode,
)
from franka_wrist_camera_scene.episode.success import pick_place_success, reaching_success


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    policy: CircleMotionPolicy | PickPlaceScriptedPolicy | ReachingScriptedPolicy,
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

    # Debug markers (only applicable for circle task)
    markers = None
    if show_markers and isinstance(policy, CircleMotionPolicy):
        markers = CircleMotionMarkers()
        points_w = circle_points_w(scene, policy.cfg, robot.device)
        markers.draw_path(points_w)

    settling = False
    settle_steps = 0
    max_settle_steps = int(1.0 / sim_dt)

    while simulation_app.is_running() and (max_steps <= 0 or step < max_steps):
        # 1. Step the policy to get reference actions
        cmd = policy.step(None, sim_time_s)

        # 2. Update and apply Cartesian IK command
        ik.set_target_pose(cmd.target_pos_w, cmd.target_quat_w)
        ik.apply(scene, robot)

        # 3. Update and apply gripper command
        gripper.set_width(cmd.finger_opening_m)
        gripper.apply(robot)

        scene.write_data_to_sim()

        sim.step()
        sim_time_s += sim_dt
        step += 1
        scene.update(sim_dt)
        probe.maybe_save(scene, step)

        if markers is not None:
            markers.draw_target(cmd.target_pos_w)

        video_recorder.record_step(scene, step)

        if cmd.done:
            if not settling:
                print(f"[INFO] Scripted policy completed execution. Settling for 1.0s ({max_settle_steps} steps)...", flush=True)
                settling = True
            settle_steps += 1
            if settle_steps >= max_settle_steps:
                if isinstance(policy, PickPlaceScriptedPolicy):
                    success = pick_place_success(scene, policy.spec)
                    print(f"[INFO] Pick-place success: {success.detach().cpu().tolist()}", flush=True)
                elif isinstance(policy, ReachingScriptedPolicy):
                    success = reaching_success(scene, policy.spec)
                    print(f"[INFO] Reaching success: {success.detach().cpu().tolist()}", flush=True)
                break

    video_recorder.close()


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        device=args_cli.device,
        physx=sim_utils.PhysxCfg(
            enable_external_forces_every_iteration=True,
            min_velocity_iteration_count=1,
            min_position_iteration_count=4,
        ),
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.2, -2.2, 1.9], target=[0.55, 0.0, 1.20])

    collection_cfg = load_yaml_config(args_cli.collection_config)
    target_object_cfg = collection_cfg["target_object"]

    import random
    seed = int(collection_cfg["seed"])
    rng = random.Random(seed)

    object_context = load_catalog_object_context(
        catalog_config=target_object_cfg["catalog_config"],
        geometry_config=target_object_cfg["geometry_config"],
        category_id=target_object_cfg["category_id"],
        variant_id=target_object_cfg["variant_id"],
        split=target_object_cfg["split"],
        role=target_object_cfg["role"],
        required_affordances=tuple(target_object_cfg["required_affordances"]),
        required_grasp_strategy=target_object_cfg["required_grasp_strategy"],
        rng=rng,
    )


    scene = InteractiveScene(
        make_tabletop_scene_cfg(
            object_context=object_context,
            num_envs=args_cli.num_envs,
            env_spacing=2.5,
        )
    )
    robot: Articulation = scene["robot"]

    # Choose policy based on selected task
    if args_cli.task == "circle":
        trajectory_cfg = CircleTrajectoryCfg(
            center_local=CIRCLE_CENTER_LOCAL,
            diameter_m=args_cli.circle_diameter,
            frequency_hz=args_cli.circle_frequency,
            orientation_wxyz=GRIPPER_DOWN_QUAT_WXYZ,
        )
        policy = CircleMotionPolicy(cfg=trajectory_cfg)
    elif args_cli.task == "reaching":
        base_spec = ReachingTaskSpec()
        spec = make_reaching_episode_spec(
            base_spec=base_spec,
            object_xy_offset=(0.0, 0.0),
            object_label=object_context.label,
        )
        policy = ReachingScriptedPolicy(spec=spec)
    else:  # pick_place
        base_spec = PickPlaceTaskSpec()
        grasp_closing_axis_xy = (
            object_context.geometry.planar_minor_axis_local
            if object_context.geometry.yaw_relevant
            else None
        )
        spec = make_pick_place_episode_spec(
            base_spec=base_spec,
            object_xy_offset=(0.0, 0.0),
            place_xy_offset=(0.0, 0.0),
            object_label=object_context.label,
            grasp_closing_axis_xy=grasp_closing_axis_xy,
        )
        policy = PickPlaceScriptedPolicy(spec=spec)

    ik = CartesianIKController()
    gripper = GripperController()
    probe = WristCameraProbe(args_cli.probe_u, args_cli.probe_v, args_cli.save_probe_every)

    sim.reset()
    policy.bind(scene, robot)
    ik.bind(scene, robot)
    gripper.bind(scene, robot)
    if args_cli.task == "pick_place":
        reset_pick_place_episode(scene, spec)
    elif args_cli.task == "reaching":
        reset_reaching_episode(scene, spec)
    else:
        reset_robot_to_default(scene)
        scene.reset()
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
