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
        "place_pos_local": tuple(meta["place_pos_local"]),
        "object_color_name": meta.get("object_color_name"),
        "object_color_rgb": tuple(meta["object_color_rgb"]) if meta.get("object_color_rgb") is not None else None,
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
        f"{'traj_steps':<10} {'meta_cam':<9} {'traj_cam':<9} {'depth':<6} {'object_color':<24} {'light':<24}"
    )

    for item in summaries:
        episode_id = f"{item['episode_id']:06d}"
        success = str(item["success"]).lower()
        record_depth = str(item["record_depth"]).lower()
        color_name = item.get("object_color_name", "none")
        color_rgb_str = (
            f"({', '.join(f'{x:.2f}' for x in item['object_color_rgb'])})"
            if item["object_color_rgb"] is not None
            else ""
        )
        color_str = f"{color_name} {color_rgb_str}".strip()
        light_str = "none"
        if item["light_intensity"] is not None and item["light_color"] is not None:
            light_color_str = f"({', '.join(f'{x:.2f}' for x in item['light_color'])})"
            light_str = f"{item['light_intensity']:.1f} {light_color_str}"
        print(
            f"{episode_id:<10} {success:<8} "
            f"{item['num_steps']:<10} {item['trajectory_steps']:<10} "
            f"{item['num_camera_frames']:<9} {item['trajectory_camera_frames']:<9} "
            f"{record_depth:<6} {color_str:<24} {light_str:<24}"
        )

    print_pose_variant_summary(summaries)


def pose_key(summary: dict) -> tuple:
    return (
        tuple(round(float(x), 4) for x in summary["object_pos_local"]),
        tuple(round(float(x), 4) for x in summary["place_pos_local"]),
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
