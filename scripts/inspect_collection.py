#!/usr/bin/env python3
"""Inspect a raw collection directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a raw tabletop collection.")
    parser.add_argument("collection_dir", type=Path)
    return parser.parse_args()


def load_episode_summary(episode_dir: Path) -> dict:
    meta_path = episode_dir / "meta.json"
    traj_path = episode_dir / "trajectory.npz"

    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    if not traj_path.exists():
        raise FileNotFoundError(traj_path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    with np.load(traj_path) as traj:
        steps = int(traj["timestamps_s"].shape[0])
        camera_frames = int(traj["camera_timestamps_s"].shape[0]) if "camera_timestamps_s" in traj.files else 0

    return {
        "episode_id": int(meta["episode_id"]),
        "success": bool(meta["success"]),
        "num_steps": int(meta["num_steps"]),
        "trajectory_steps": steps,
        "num_camera_frames": int(meta.get("num_camera_frames", camera_frames)),
        "trajectory_camera_frames": camera_frames,
        "record_depth": bool(meta.get("record_depth", False)),
        "object_pos_local": tuple(meta["object_pos_local"]),
        "place_pos_local": tuple(meta["place_pos_local"]) if meta["place_pos_local"] is not None else None,
        "object_category_id": meta.get("object_category_id"),
        "object_variant_id": meta.get("object_variant_id"),
        "object_label": meta.get("object_label"),
        "object_usd_path": meta.get("object_usd_path"),
        "object_grasp_strategy": meta.get("object_grasp_strategy"),
        "object_yaw_relevant": meta["object_yaw_relevant"],
        "object_planar_aspect_ratio": meta["object_planar_aspect_ratio"],
        "object_planar_minor_axis_local": meta["object_planar_minor_axis_local"],
        "grasp_closing_axis_xy": meta["grasp_closing_axis_xy"],
        "placement_target_category_id": meta.get("placement_target_category_id"),
        "placement_target_variant_id": meta.get("placement_target_variant_id"),
        "placement_target_label": meta.get("placement_target_label"),
        "placement_target_usd_path": meta.get("placement_target_usd_path"),
        "placement_target_grasp_strategy": meta.get("placement_target_grasp_strategy"),
        "placement_target_pos_local": (
            tuple(meta["placement_target_pos_local"])
            if meta.get("placement_target_pos_local") is not None
            else None
        ),
        "light_intensity": meta.get("light_intensity"),
        "light_color": tuple(meta["light_color"]) if meta.get("light_color") is not None else None,
        "clutter_objects": meta.get("clutter_objects"),
    }


def format_clutter_summary(clutter_objects: list[dict] | None) -> str:
    if clutter_objects is None:
        return "missing"
    labels = [
        f"{item['category_id']}/{item['variant_id']}"
        for item in clutter_objects
    ]
    return f"{len(clutter_objects)} [{', '.join(labels)}]"


def main() -> None:
    args = parse_args()
    collection_dir: Path = args.collection_dir

    manifest_path = collection_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episode_dirs = [collection_dir / item["episode_dir"] for item in manifest["episodes"]]

    summaries = [load_episode_summary(path) for path in episode_dirs]
    successes = sum(item["success"] for item in summaries)

    if any(item.get("clutter_objects") is None for item in summaries):
        print("[WARN] Some episodes are missing clutter_objects metadata.", flush=True)

    print(f"collection: {collection_dir}")
    print(f"episodes: {len(summaries)}")
    print(f"success: {successes}/{len(summaries)}")
    print()
    for item in summaries:
        episode_id = f"{item['episode_id']:06d}"
        success = str(item["success"]).lower()
        object_variant = item.get("object_variant_id", "none") or "none"
        object_ref = f"{item.get('object_category_id', 'none')}/{object_variant}"
        receptacle_ref = (
            f"{item.get('placement_target_category_id', 'none')}/"
            f"{item.get('placement_target_variant_id', 'none')}"
        )
        clutter_summary = format_clutter_summary(item.get("clutter_objects"))
        print(
            f"episode={episode_id} success={success} "
            f"object={object_ref} receptacle={receptacle_ref} "
            f"clutter={clutter_summary}"
        )

    print_pose_variant_summary(summaries)


def pose_key(summary: dict) -> tuple:
    place_pos_local = summary["place_pos_local"]
    placement_target_pos_local = summary["placement_target_pos_local"]
    return (
        tuple(round(float(x), 4) for x in summary["object_pos_local"]),
        tuple(round(float(x), 4) for x in place_pos_local) if place_pos_local is not None else None,
        (
            tuple(round(float(x), 4) for x in placement_target_pos_local)
            if placement_target_pos_local is not None
            else None
        ),
    )


def print_pose_variant_summary(summaries: list[dict]) -> None:
    grouped: dict[tuple, list[dict]] = {}

    for item in summaries:
        grouped.setdefault(pose_key(item), []).append(item)

    print()
    print("success by pose variant:")
    print(f"{'object_pos_local':<26} {'place_pos_local':<26} {'placement_pos_local':<26} {'success':<8}")

    for (object_pos, place_pos, placement_pos), items in sorted(grouped.items()):
        successes = sum(item["success"] for item in items)
        total = len(items)
        print(f"{str(object_pos):<26} {str(place_pos):<26} {str(placement_pos):<26} {successes}/{total:<8}")


if __name__ == "__main__":
    main()
