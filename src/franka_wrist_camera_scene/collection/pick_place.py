"""Pick-and-place data collection orchestration pipeline."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene

from franka_wrist_camera_scene.control.gripper import GripperController
from franka_wrist_camera_scene.control.ik import CartesianIKController
from franka_wrist_camera_scene.episode.reset import reset_pick_place_episode
from franka_wrist_camera_scene.episode.success import pick_place_success
from franka_wrist_camera_scene.episode.recorder import EpisodeRecorder
from franka_wrist_camera_scene.policies.pick_place_scripted import PickPlaceScriptedPolicy
from franka_wrist_camera_scene.scene.tabletop import TabletopFrankaSceneCfg, CATALOG_OBJECT_LABEL
from franka_wrist_camera_scene.settings import SIM_DT
from franka_wrist_camera_scene.tasks.pick_place import PickPlaceTaskSpec, make_pick_place_episode_spec
from franka_wrist_camera_scene.app.camera_warmup import nudge_camera_prims
from franka_wrist_camera_scene.episode.manifest import write_collection_manifest
from franka_wrist_camera_scene.tasks.sampling import (
    parse_xy_range,
    sample_pick_place_offsets,
    parse_lighting_options,
)
from franka_wrist_camera_scene.scene.lighting import set_dome_light


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
    simulation_app,
    seed: int | None = None,
    object_xy_offset: tuple[float, float] | None = None,
    place_xy_offset: tuple[float, float] | None = None,
    object_color_name: str | None = None,
    object_color_rgb: tuple[float, float, float] | None = None,
    light_intensity: float | None = None,
    light_color: tuple[float, float, float] | None = None,
) -> Path:
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
        object_pos_local=policy.spec.object_pos_local,
        place_pos_local=policy.spec.place_pos_local,
        seed=seed,
        object_xy_offset=object_xy_offset,
        place_xy_offset=place_xy_offset,
        object_color_name=object_color_name,
        object_color_rgb=object_color_rgb,
        light_intensity=light_intensity,
        light_color=light_color,
    )
    recorder.validate_output_path()

    settling = False
    settle_steps = 0
    max_settle_steps = int(settle_time_s / sim_dt)
    completed = False

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
                print(
                    f"[INFO] Scripted policy completed execution. Settling for {settle_time_s}s ({max_settle_steps} steps)...",
                    flush=True,
                )
                settling = True
            settle_steps += 1
            if settle_steps >= max_settle_steps:
                completed = True
                break

    if not completed:
        if step >= max_steps:
            raise RuntimeError(f"Episode exceeded max_steps={max_steps} before policy completion.")
        raise RuntimeError("Simulation stopped before episode completion.")

    # Check success
    success = bool(pick_place_success(scene, policy.spec)[0].item())
    print(f"[INFO] Episode {episode_id} success: {success}", flush=True)

    # Save episode data
    saved_dir = recorder.save(success)
    print(f"[INFO] Saved episode data to: {saved_dir}", flush=True)
    return saved_dir


def collect_pick_place_dataset(
    collection_cfg: dict,
    device: str,
    simulation_app,
) -> None:
    """Run the pick-and-place data collection pipeline."""
    sim_cfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        device=device,
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

    ik = CartesianIKController()
    gripper = GripperController()

    sim.reset()
    ik.bind(scene, robot)
    gripper.bind(scene, robot)

    seed = int(collection_cfg["seed"])
    pose_randomization = collection_cfg["pose_randomization"]
    object_xy_range = parse_xy_range(pose_randomization["object_xy_range"])
    place_xy_range = parse_xy_range(pose_randomization["place_xy_range"])

    lighting_randomization = collection_cfg["lighting_randomization"]
    lighting_options = parse_lighting_options(lighting_randomization)

    output_dir = Path(collection_cfg["output_dir"])
    start_episode_id = int(collection_cfg["start_episode_id"])
    num_episodes = int(collection_cfg["num_episodes"])
    max_steps = int(collection_cfg["max_steps"])
    settle_time_s = float(collection_cfg["settle_time_s"])
    record_cameras = bool(collection_cfg["record_cameras"])
    record_depth = bool(collection_cfg.get("record_depth", False))
    camera_fps = int(collection_cfg.get("camera_fps", 30))

    saved_episode_dirs: list[Path] = []

    for episode_id in range(start_episode_id, start_episode_id + num_episodes):
        print(f"[INFO] Starting episode {episode_id}", flush=True)
        sample = sample_pick_place_offsets(
            seed=seed,
            episode_id=episode_id,
            object_range=object_xy_range,
            place_range=place_xy_range,
            lighting=lighting_options,
        )
        episode_spec = make_pick_place_episode_spec(
            base_spec=spec,
            object_xy_offset=sample.object_xy_offset,
            place_xy_offset=sample.place_xy_offset,
            object_label=CATALOG_OBJECT_LABEL,
        )

        policy = PickPlaceScriptedPolicy(spec=episode_spec)
        policy.bind(scene, robot)

        reset_pick_place_episode(scene, episode_spec)
        # USD catalog objects keep their authored materials.
        set_dome_light(scene, sample.light_intensity, sample.light_color)
        policy.reset()
        ik.reset()
        nudge_camera_prims(sim, scene)

        saved_dir = run_episode(
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
            simulation_app=simulation_app,
            seed=seed,
            object_xy_offset=sample.object_xy_offset,
            place_xy_offset=sample.place_xy_offset,
            light_intensity=sample.light_intensity,
            light_color=sample.light_color,
        )
        saved_episode_dirs.append(saved_dir)

    manifest_path = write_collection_manifest(
        output_dir=output_dir,
        task_name="pick_place",
        episode_dirs=saved_episode_dirs,
    )
    print(f"[INFO] Saved collection manifest to: {manifest_path}", flush=True)
