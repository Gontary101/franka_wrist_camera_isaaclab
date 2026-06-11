#!/usr/bin/env python3
"""Inspect a recorded raw episode's metadata and trajectory arrays."""

import argparse
import json
from pathlib import Path
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a recorded episode.")
    parser.add_argument(
        "--episode_dir",
        type=str,
        default="data/raw/debug_pick_place/000000",
        help="Path to the episode directory (containing meta.json and trajectory.npz).",
    )
    args = parser.parse_args()

    episode_path = Path(args.episode_dir)
    if not episode_path.exists():
        print(f"Error: Episode directory does not exist: {episode_path}")
        return

    meta_path = episode_path / "meta.json"
    traj_path = episode_path / "trajectory.npz"

    print("=" * 60)
    print(f"Inspecting Episode Directory: {episode_path.resolve()}")
    print("=" * 60)

    # 1. Inspect meta.json
    if meta_path.exists():
        print("\n--- Metadata (meta.json) ---")
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print(json.dumps(meta, indent=2))
        except Exception as e:
            print(f"Error reading meta.json: {e}")
    else:
        print("\nWarning: meta.json does not exist!")

    # 2. Inspect trajectory.npz
    if traj_path.exists():
        print("\n--- Trajectory Arrays (trajectory.npz) ---")
        try:
            with np.load(traj_path) as data:
                for key in sorted(data.files):
                    array = data[key]
                    print(f"  {key:<25} shape: {str(array.shape):<20} dtype: {str(array.dtype):<10}")
        except Exception as e:
            print(f"Error reading trajectory.npz: {e}")
    else:
        print("\nWarning: trajectory.npz does not exist!")
    print("=" * 60)


if __name__ == "__main__":
    main()
