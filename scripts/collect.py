#!/usr/bin/env python3
"""Collect deterministic pick-place episodes in the tabletop scene."""

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
        "--collection_config",
        type=str,
        default="collection.yaml",
        help="Collection config file under configs/.",
    )
    # Add app launcher arguments
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.kit_args = f"{args.kit_args} --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false".strip()
    return args


args_cli = parse_args()

# Load collection config from config folder
from franka_wrist_camera_scene.utils.paths import load_yaml_config  # noqa: E402
collection_cfg = load_yaml_config(args_cli.collection_config)

# Fail fast if any of the output directories in the range already exist
output_dir = Path(collection_cfg["output_dir"])
start_episode_id = int(collection_cfg["start_episode_id"])
num_episodes = int(collection_cfg["num_episodes"])

for episode_id in range(start_episode_id, start_episode_id + num_episodes):
    episode_dir = output_dir / f"{episode_id:06d}"
    if episode_dir.exists():
        raise FileExistsError(f"Episode directory already exists: {episode_dir}")

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
    episode_id: int,
    max_steps: int,
    settle_time_s: float,
    record_cameras: bool,
    record_depth: bool,
    camera_fps: int,
) -> None:
    """Run one episode, record data, check success, and save."""
    robot: Articulation = scene["robot"]
    sim_dt = sim.get_physics_dt()
    sim_time_s = 0.0
    step = 0

    camera_interval_steps = max(1, round(1.0 / (camera_fps * sim_dt)))

    # Initialize EpisodeRecorder
    recorder = EpisodeRecorder(
        output_dir=output_dir,
        episode_id=episode_id,
        task_name="pick_place",
        instruction=policy.spec.instruction,
        sim_dt=sim_dt,
        ee_body_id=ik.end_effector_body_id,
        object_name=policy.spec.object_name,
        record_cameras=record_cameras,
        record_depth=record_depth,
    )
    recorder.validate_output_path()

    settling = False
    settle_steps = 0
    max_settle_steps = int(settle_time_s / sim_dt)

    while simulation_app.is_running() and step < max_steps:
        # 1. Step the policy to get reference actions
        cmd = policy.step(None, sim_time_s)

        # 2. Update and apply Cartesian IK command
        ik.set_target_pose(cmd.target_pos_w, cmd.target_quat_w)
        ik.apply(scene, robot)

        # 3. Update and apply gripper command
        gripper.set_width(cmd.finger_opening_m)
        gripper.apply(robot)

        scene.write_data_to_sim()

        # Dataset convention: record state_t and command_t before advancing to state_{t+1}.
        recorder.record_step(scene, cmd, step, sim_time_s)

        if record_cameras and step % camera_interval_steps == 0:
            recorder.record_cameras_step(scene, step, sim_time_s)

        sim.step()
        sim_time_s += sim_dt
        step += 1
        scene.update(sim_dt)

        if cmd.done:
            if not settling:
                print(f"[INFO] Scripted policy completed execution. Settling for {settle_time_s}s ({max_settle_steps} steps)...", flush=True)
                settling = True
            settle_steps += 1
            if settle_steps >= max_settle_steps:
                break

    if step >= max_steps:
        raise RuntimeError(f"Episode exceeded max_steps={max_steps} before policy completion.")

    # Check success
    success = bool(pick_place_success(scene, policy.spec)[0].item())
    print(f"[INFO] Episode {episode_id} success: {success}", flush=True)

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

    output_dir = Path(collection_cfg["output_dir"])
    start_episode_id = int(collection_cfg["start_episode_id"])
    num_episodes = int(collection_cfg["num_episodes"])
    max_steps = int(collection_cfg["max_steps"])
    settle_time_s = float(collection_cfg["settle_time_s"])
    record_cameras = bool(collection_cfg["record_cameras"])
    record_depth = bool(collection_cfg.get("record_depth", False))
    camera_fps = int(collection_cfg.get("camera_fps", 30))

    for episode_id in range(start_episode_id, start_episode_id + num_episodes):
        print(f"[INFO] Starting episode {episode_id}", flush=True)
        reset_pick_place_episode(scene, spec)
        policy.reset()
        ik.reset()
        nudge_camera_prims(sim, scene)

        run_episode(
            sim=sim,
            scene=scene,
            policy=policy,
            ik=ik,
            gripper=gripper,
            output_dir=output_dir,
            episode_id=episode_id,
            max_steps=max_steps,
            settle_time_s=settle_time_s,
            record_cameras=record_cameras,
            record_depth=record_depth,
            camera_fps=camera_fps,
        )


if __name__ == "__main__":
    main()
    simulation_app.close()
