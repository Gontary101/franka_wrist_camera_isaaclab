"""Pick-and-place data collection orchestration pipeline."""

from __future__ import annotations

import gc
from pathlib import Path
import random

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.scene import InteractiveScene

from franka_wrist_camera_scene.app.camera_warmup import nudge_camera_prims
from franka_wrist_camera_scene.app.stage_lifecycle import delete_scene_prims
from franka_wrist_camera_scene.control.gripper import GripperController
from franka_wrist_camera_scene.control.ik import CartesianIKController
from franka_wrist_camera_scene.episode.manifest import write_collection_manifest
from franka_wrist_camera_scene.episode.recorder import EpisodeRecorder
from franka_wrist_camera_scene.episode.reset import reset_pick_place_episode
from franka_wrist_camera_scene.episode.success import pick_place_success
from franka_wrist_camera_scene.policies.pick_place_scripted import PickPlaceScriptedPolicy
from franka_wrist_camera_scene.scene.lighting import set_dome_light
from franka_wrist_camera_scene.scene.object_context import CatalogObjectContext, load_catalog_object_context
from franka_wrist_camera_scene.scene.tabletop import make_pick_place_tabletop_scene_cfg
from franka_wrist_camera_scene.settings import SIM_DT
from franka_wrist_camera_scene.tasks.placement_geometry import object_root_pose_on_support
from franka_wrist_camera_scene.tasks.pick_place import PickPlaceTaskSpec, make_pick_place_episode_spec
from franka_wrist_camera_scene.tasks.sampling import (
    parse_lighting_options,
    parse_xy_range,
    sample_pick_place_offsets,
)
from franka_wrist_camera_scene.utils.paths import REPO_ROOT


def _load_collection_object_context(sampling_cfg: dict, rng: random.Random) -> CatalogObjectContext:
    return load_catalog_object_context(
        catalog_config=sampling_cfg["catalog_config"],
        geometry_config=sampling_cfg["geometry_config"],
        category_id=sampling_cfg["category_id"],
        variant_id=sampling_cfg["variant_id"],
        split=sampling_cfg["split"],
        role=sampling_cfg["role"],
        required_affordances=tuple(sampling_cfg["required_affordances"]),
        required_grasp_strategy=sampling_cfg["required_grasp_strategy"],
        rng=rng,
    )


def _repo_relative_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


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
    object_category_id: str | None = None,
    object_variant_id: str | None = None,
    object_label: str | None = None,
    object_usd_path: str | None = None,
    object_grasp_strategy: str | None = None,
    object_yaw_relevant: bool | None = None,
    object_planar_aspect_ratio: float | None = None,
    object_planar_minor_axis_local: tuple[float, float] | None = None,
    object_planar_major_axis_local: tuple[float, float] | None = None,
    grasp_closing_axis_xy: tuple[float, float] | None = None,
    placement_target_category_id: str | None = None,
    placement_target_variant_id: str | None = None,
    placement_target_label: str | None = None,
    placement_target_usd_path: str | None = None,
    placement_target_grasp_strategy: str | None = None,
    placement_target_pos_local: tuple[float, float, float] | None = None,
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
        object_category_id=object_category_id,
        object_variant_id=object_variant_id,
        object_label=object_label,
        object_usd_path=object_usd_path,
        object_grasp_strategy=object_grasp_strategy,
        object_yaw_relevant=object_yaw_relevant,
        object_planar_aspect_ratio=object_planar_aspect_ratio,
        object_planar_minor_axis_local=object_planar_minor_axis_local,
        object_planar_major_axis_local=object_planar_major_axis_local,
        grasp_closing_axis_xy=grasp_closing_axis_xy,
        placement_target_category_id=placement_target_category_id,
        placement_target_variant_id=placement_target_variant_id,
        placement_target_label=placement_target_label,
        placement_target_usd_path=placement_target_usd_path,
        placement_target_grasp_strategy=placement_target_grasp_strategy,
        placement_target_pos_local=placement_target_pos_local,
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
                    "[INFO] Scripted policy completed execution. "
                    f"Settling for {settle_time_s}s ({max_settle_steps} steps)...",
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

    target_object_cfg = collection_cfg["target_object"]
    placement_target_cfg = collection_cfg["placement_target"]

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
    record_depth = bool(collection_cfg["record_depth"])
    camera_fps = int(collection_cfg["camera_fps"])

    saved_episode_dirs: list[Path] = []

    for episode_id in range(start_episode_id, start_episode_id + num_episodes):
        print(f"[INFO] Starting episode {episode_id}", flush=True)
        scene = None
        robot = None
        ik = None
        gripper = None
        policy = None

        try:
            target_rng = random.Random(seed + episode_id)
            placement_rng = random.Random(seed + 100_000 + episode_id)
            object_context = _load_collection_object_context(target_object_cfg, target_rng)
            placement_context = _load_collection_object_context(placement_target_cfg, placement_rng)
            object_usd_path = _repo_relative_path(object_context.usd_path)
            placement_usd_path = _repo_relative_path(placement_context.usd_path)

            spec = PickPlaceTaskSpec()
            sample = sample_pick_place_offsets(
                seed=seed,
                episode_id=episode_id,
                object_range=object_xy_range,
                place_range=place_xy_range,
                lighting=lighting_options,
            )
            placement_xy = (
                spec.place_pos_local[0] + sample.place_xy_offset[0],
                spec.place_pos_local[1] + sample.place_xy_offset[1],
            )
            placement_receptacle_pos_local = object_root_pose_on_support(
                xy_pos=placement_xy,
                support_surface_z=spec.support_surface_z_local,
                object_bbox_min_z=placement_context.geometry.local_bbox_min[2],
                bottom_clearance_m=spec.object_bottom_clearance_m,
            )

            scene = InteractiveScene(
                make_pick_place_tabletop_scene_cfg(
                    object_context=object_context,
                    placement_context=placement_context,
                    placement_pos_local=placement_receptacle_pos_local,
                    num_envs=1,
                    env_spacing=2.5,
                )
            )
            robot: Articulation = scene["robot"]

            ik = CartesianIKController()
            gripper = GripperController()

            sim.reset()
            sim.set_camera_view(eye=[2.2, -2.2, 1.9], target=[0.55, 0.0, 1.20])
            ik.bind(scene, robot)
            gripper.bind(scene, robot)

            grasp_closing_axis_xy = (
                object_context.geometry.planar_minor_axis_local
                if object_context.geometry.yaw_relevant
                else None
            )
            episode_spec = make_pick_place_episode_spec(
                base_spec=spec,
                object_xy_offset=sample.object_xy_offset,
                place_xy_offset=sample.place_xy_offset,
                object_label=object_context.label,
                grasp_closing_axis_xy=grasp_closing_axis_xy,
                object_local_bbox_min=object_context.geometry.local_bbox_min,
                object_local_bbox_max=object_context.geometry.local_bbox_max,
                placement_target_pos_local=placement_receptacle_pos_local,
                placement_target_local_bbox_min=placement_context.geometry.local_bbox_min,
                placement_target_local_bbox_max=placement_context.geometry.local_bbox_max,
                placement_label=placement_context.label,
            )

            policy = PickPlaceScriptedPolicy(spec=episode_spec)
            policy.bind(scene, robot)

            reset_pick_place_episode(scene, episode_spec)
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
                object_category_id=object_context.category_id,
                object_variant_id=object_context.variant_id,
                object_label=object_context.label,
                object_usd_path=object_usd_path,
                object_grasp_strategy=object_context.grasp_strategy,
                object_yaw_relevant=object_context.geometry.yaw_relevant,
                object_planar_aspect_ratio=object_context.geometry.planar_aspect_ratio,
                object_planar_minor_axis_local=object_context.geometry.planar_minor_axis_local,
                object_planar_major_axis_local=object_context.geometry.planar_major_axis_local,
                grasp_closing_axis_xy=episode_spec.grasp_closing_axis_xy,
                placement_target_category_id=placement_context.category_id,
                placement_target_variant_id=placement_context.variant_id,
                placement_target_label=placement_context.label,
                placement_target_usd_path=placement_usd_path,
                placement_target_grasp_strategy=placement_context.grasp_strategy,
                placement_target_pos_local=placement_receptacle_pos_local,
                light_intensity=sample.light_intensity,
                light_color=sample.light_color,
            )
            saved_episode_dirs.append(saved_dir)
        finally:
            del scene, robot, ik, gripper, policy
            gc.collect()
            delete_scene_prims(sim=sim, simulation_app=simulation_app)

    manifest_path = write_collection_manifest(
        output_dir=output_dir,
        task_name="pick_place",
        episode_dirs=saved_episode_dirs,
    )
    print(f"[INFO] Saved collection manifest to: {manifest_path}", flush=True)
