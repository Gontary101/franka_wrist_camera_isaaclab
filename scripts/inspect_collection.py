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
        "light_intensity": meta.get("light_intensity"),
        "light_color": tuple(meta["light_color"]) if meta.get("light_color") is not None else None,
    }


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

    print(f"collection: {collection_dir}")
    print(f"episodes: {len(summaries)}")
    print(f"success: {successes}/{len(summaries)}")
    print()
    print(
        f"{'episode_id':<10} {'success':<8} {'meta_steps':<10} "
        f"{'traj_steps':<10} {'meta_cam':<9} {'traj_cam':<9} {'depth':<6} "
        f"{'object_variant':<20} {'strategy':<12} {'yaw':<6} {'aspect':<8} "
        f"{'minor_axis':<20} {'grasp_axis':<20} {'light':<24}"
    )

    for item in summaries:
        episode_id = f"{item['episode_id']:06d}"
        success = str(item["success"]).lower()
        record_depth = str(item["record_depth"]).lower()
        variant_id = item.get("object_variant_id", "none") or "none"
        strategy = item.get("object_grasp_strategy", "none") or "none"
        light_str = "none"
        if item["light_intensity"] is not None and item["light_color"] is not None:
            light_color_str = f"({', '.join(f'{x:.2f}' for x in item['light_color'])})"
            light_str = f"{item['light_intensity']:.1f} {light_color_str}"
        yaw_relevant = str(item["object_yaw_relevant"]).lower()
        aspect_ratio = item["object_planar_aspect_ratio"]
        aspect_str = f"{aspect_ratio:.3f}" if aspect_ratio is not None else "none"
        minor_axis = item["object_planar_minor_axis_local"]
        minor_axis_str = str(minor_axis) if minor_axis is not None else "none"
        grasp_axis = item["grasp_closing_axis_xy"]
        grasp_axis_str = str(grasp_axis) if grasp_axis is not None else "none"
        print(
            f"{episode_id:<10} {success:<8} "
            f"{item['num_steps']:<10} {item['trajectory_steps']:<10} "
            f"{item['num_camera_frames']:<9} {item['trajectory_camera_frames']:<9} "
            f"{record_depth:<6} {variant_id:<20} {strategy:<12} {yaw_relevant:<6} "
            f"{aspect_str:<8} {minor_axis_str:<20} {grasp_axis_str:<20} {light_str:<24}"
        )

    print_pose_variant_summary(summaries)


def pose_key(summary: dict) -> tuple:
    place_pos_local = summary["place_pos_local"]
    return (
        tuple(round(float(x), 4) for x in summary["object_pos_local"]),
        tuple(round(float(x), 4) for x in place_pos_local) if place_pos_local is not None else None,
    )


def print_pose_variant_summary(summaries: list[dict]) -> None:
    grouped: dict[tuple, list[dict]] = {}

    for item in summaries:
        grouped.setdefault(pose_key(item), []).append(item)

    print()
    print("success by pose variant:")
    print(f"{'object_pos_local':<26} {'place_pos_local':<26} {'success':<8}")

    for (object_pos, place_pos), items in sorted(grouped.items()):
        successes = sum(item["success"] for item in items)
        total = len(items)
        print(f"{str(object_pos):<26} {str(place_pos):<26} {successes}/{total:<8}")


if __name__ == "__main__":
    main()
