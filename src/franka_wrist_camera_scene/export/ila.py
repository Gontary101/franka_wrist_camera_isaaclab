"""Exporter for image-language-action datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def export_episode(
    raw_collection_dir: Path,
    export_dir: Path,
    episode_entry: dict,
) -> dict:
    episode_id = int(episode_entry["episode_id"])
    raw_meta_path = raw_collection_dir / episode_entry["metadata_file"]
    raw_traj_path = raw_collection_dir / episode_entry["trajectory_file"]

    meta = load_json(raw_meta_path)

    with np.load(raw_traj_path) as traj:
        idx = traj["camera_step_indices"].astype(np.int64)
        ee_pos_w = traj["ee_pos_w"][idx]
        action_target_pos_w = traj["action_target_pos_w"][idx]
        delta_target_pos_w = action_target_pos_w - ee_pos_w

        arrays = {
            "agent_rgb": traj["agent_rgb"],
            "wrist_rgb": traj["wrist_rgb"],
            "ee_pos_w": ee_pos_w,
            "object_pos_w": traj["object_pos_w"][idx],
            "action_target_pos_w": action_target_pos_w,
            "action_target_quat_w": traj["action_target_quat_w"][idx],
            "action_delta_target_pos_w": delta_target_pos_w,
            "action_finger_opening_m": traj["action_finger_opening_m"][idx],
            "timestamps_s": traj["camera_timestamps_s"],
            "source_control_step_indices": idx,
        }

        if "agent_depth" in traj.files and "wrist_depth" in traj.files:
            arrays["agent_depth"] = traj["agent_depth"]
            arrays["wrist_depth"] = traj["wrist_depth"]

        episode_file = export_dir / "episodes" / f"{episode_id:06d}.npz"
        np.savez_compressed(episode_file, **arrays)

    return {
        "episode_id": episode_id,
        "episode_file": f"episodes/{episode_id:06d}.npz",
        "source_episode_dir": episode_entry["episode_dir"],
        "instruction": meta["instruction"],
        "success": bool(meta["success"]),
        "num_frames": int(arrays["timestamps_s"].shape[0]),
        "object_pos_local": meta["object_pos_local"],
        "place_pos_local": meta["place_pos_local"],
        "object_category_id": meta.get("object_category_id"),
        "object_variant_id": meta.get("object_variant_id"),
        "object_label": meta.get("object_label"),
        "object_usd_path": meta.get("object_usd_path"),
        "object_grasp_strategy": meta.get("object_grasp_strategy"),
        "object_yaw_relevant": meta["object_yaw_relevant"],
        "object_planar_aspect_ratio": meta["object_planar_aspect_ratio"],
        "object_planar_minor_axis_local": meta["object_planar_minor_axis_local"],
        "object_planar_major_axis_local": meta["object_planar_major_axis_local"],
        "grasp_closing_axis_xy": meta["grasp_closing_axis_xy"],
        "placement_target_category_id": meta.get("placement_target_category_id"),
        "placement_target_variant_id": meta.get("placement_target_variant_id"),
        "placement_target_label": meta.get("placement_target_label"),
        "placement_target_usd_path": meta.get("placement_target_usd_path"),
        "placement_target_grasp_strategy": meta.get("placement_target_grasp_strategy"),
        "placement_target_pos_local": meta.get("placement_target_pos_local"),
        "light_intensity": meta.get("light_intensity"),
        "light_color": meta.get("light_color"),
    }


def export_collection_to_ila(
    raw_collection_dir: Path,
    export_dir: Path,
) -> Path:
    raw_manifest_path = raw_collection_dir / "manifest.json"
    raw_manifest = load_json(raw_manifest_path)

    episodes_dir = export_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=False)

    exported_episodes = [
        export_episode(raw_collection_dir, export_dir, episode_entry)
        for episode_entry in raw_manifest["episodes"]
    ]

    manifest = {
        "format_version": 1,
        "dataset_type": "image_language_action",
        "source_collection": str(raw_collection_dir),
        "task_name": raw_manifest["task_name"],
        "num_episodes": len(exported_episodes),
        "camera_names": ["agent_rgb", "wrist_rgb"],
        "action_space": "relative_cartesian_target_plus_gripper",
        "action_keys": [
            "action_delta_target_pos_w",
            "action_target_quat_w",
            "action_finger_opening_m",
        ],
        "state_keys": [
            "ee_pos_w",
            "object_pos_w",
        ],
        "observation_keys": [
            "agent_rgb",
            "wrist_rgb",
        ],
        "episodes": exported_episodes,
    }

    manifest_path = export_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path
