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
        "record_cameras": bool(meta.get("record_cameras", False)),
        "record_depth": bool(meta.get("record_depth", False)),
    }


def main() -> None:
    args = parse_args()
    collection_dir: Path = args.collection_dir

    episode_dirs = sorted(path for path in collection_dir.iterdir() if path.is_dir())
    if not episode_dirs:
        raise RuntimeError(f"No episode directories found in {collection_dir}")

    summaries = [load_episode_summary(path) for path in episode_dirs]
    successes = sum(item["success"] for item in summaries)

    print(f"collection: {collection_dir}")
    print(f"episodes: {len(summaries)}")
    print(f"success: {successes}/{len(summaries)}")
    print()
    print(f"{'episode_id':<10} {'success':<8} {'steps':<7} {'camera_frames':<14} {'record_depth':<12}")

    for item in summaries:
        episode_id = f"{item['episode_id']:06d}"
        success = str(item["success"]).lower()
        steps = item["num_steps"]
        camera_frames = item["num_camera_frames"]
        record_depth = str(item["record_depth"]).lower()
        print(f"{episode_id:<10} {success:<8} {steps:<7} {camera_frames:<14} {record_depth:<12}")


if __name__ == "__main__":
    main()
