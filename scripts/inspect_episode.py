#!/usr/bin/env python3
"""Inspect one raw recorded episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one raw tabletop episode.")
    parser.add_argument("episode_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    episode_dir: Path = args.episode_dir

    meta_path = episode_dir / "meta.json"
    traj_path = episode_dir / "trajectory.npz"

    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    if not traj_path.exists():
        raise FileNotFoundError(traj_path)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    traj = np.load(traj_path)

    print("metadata:")
    for key, value in meta.items():
        print(f"  {key}: {value}")

    print("\ntrajectory.npz:")
    for key in traj.files:
        array = traj[key]
        print(f"  {key:<28} {str(array.shape):<24} {array.dtype}")


if __name__ == "__main__":
    main()
