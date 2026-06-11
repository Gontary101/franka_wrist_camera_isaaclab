#!/usr/bin/env python3
"""Collect a single deterministic pick-place episode in the tabletop scene."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

# Import launcher to apply Isaac Sim 6.0 and pxr compatibility patches before importing isaaclab
from franka_wrist_camera_scene.app import launcher  # noqa: F401
from isaaclab.app import AppLauncher  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect deterministic pick-and-place tabletop episodes.")
    parser.add_argument(
        "--output_dir", type=str, default="data/raw/debug_pick_place", help="Output directory to save raw data."
    )
    # Add app launcher arguments
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

from franka_wrist_camera_scene.control.gripper import GripperController  # noqa: E402
from franka_wrist_camera_scene.control.ik import CartesianIKController  # noqa: E402
from franka_wrist_camera_scene.episode.reset import reset_pick_place_episode  # noqa: E402
from franka_wrist_camera_scene.episode.success import pick_place_success  # noqa: E402
from franka_wrist_camera_scene.episode.recorder import EpisodeRecorder  # noqa: E402
from franka_wrist_camera_scene.policies.pick_place_scripted import PickPlaceScriptedPolicy  # noqa: E402
from franka_wrist_camera_scene.scene.tabletop import TabletopFrankaSceneCfg  # noqa: E402
from franka_wrist_camera_scene.settings import SIM_DT  # noqa: E402
from franka_wrist_camera_scene.tasks.pick_place import PickPlaceTaskSpec  # noqa: E402
from franka_wrist_camera_scene.app.camera_warmup import nudge_camera_prims  # noqa: E402


def run_episode(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    policy: PickPlaceScriptedPolicy,
    ik: CartesianIKController,
    gripper: GripperController,
    output_dir: Path,
) -> None:
    """Run one episode, record data, check success, and save."""
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()
    sim_time_s = 0.0
    step = 0

    # Initialize EpisodeRecorder
    recorder = EpisodeRecorder(
        output_dir=output_dir,
        episode_id=0,
        task_name="pick_place",
        instruction=policy.spec.instruction,
        sim_dt=sim_dt,
        ee_body_id=ik.end_effector_body_id,
        object_name=policy.spec.object_name,
    )

    settling = False
    settle_steps = 0
    max_settle_steps = int(1.0 / sim_dt)

    while simulation_app.is_running():
        # 1. Step the policy to get reference actions
        cmd = policy.step(None, sim_time_s)

        # 2. Update and apply Cartesian IK command
        ik.set_target_pose(cmd.target_pos_w, cmd.target_quat_w)
        ik.apply(scene, robot)

        # 3. Update and apply gripper command
        gripper.set_width(cmd.finger_opening_m)
        gripper.apply(robot)

        scene.write_data_to_sim()

        # Record control step
        recorder.record_step(scene, cmd)

        sim.step()
        sim_time_s += sim_dt
        step += 1
        scene.update(sim_dt)

        if cmd.done:
            if not settling:
                print(f"[INFO] Scripted policy completed execution. Settling for 1.0s ({max_settle_steps} steps)...", flush=True)
                settling = True
            settle_steps += 1
            if settle_steps >= max_settle_steps:
                break

    # Check success
    success = bool(pick_place_success(scene, policy.spec)[0].item())
    print(f"[INFO] Episode 0 success: {success}", flush=True)

    # Save episode data
    saved_dir = recorder.save(success)
    print(f"[INFO] Saved episode data to: {saved_dir}", flush=True)


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

    scene = InteractiveScene(TabletopFrankaSceneCfg(num_envs=1, env_spacing=2.5))
    robot: Articulation = scene["robot"]

    spec = PickPlaceTaskSpec()
    policy = PickPlaceScriptedPolicy(spec=spec)

    ik = CartesianIKController()
    gripper = GripperController()

    sim.reset()
    policy.bind(scene, robot)
    ik.bind(scene, robot)
    gripper.bind(scene, robot)

    # Clean episode reset
    reset_pick_place_episode(scene, spec)
    ik.reset()

    nudge_camera_prims(sim, scene)

    output_dir = Path(args_cli.output_dir)
    run_episode(sim, scene, policy, ik, gripper, output_dir)


if __name__ == "__main__":
    main()
    simulation_app.close()
