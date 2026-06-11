#!/usr/bin/env python3
"""Inspect an exported image-language-action dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from franka_wrist_camera_scene.datasets.ila import ILADataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect an exported ILA dataset.")
    parser.add_argument("dataset_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ILADataset(args.dataset_dir)

    print(f"dataset: {args.dataset_dir}")
    print(f"episodes: {len(dataset.episodes)}")
    print(f"frames: {len(dataset)}")
    print(f"observation_keys: {dataset.observation_keys}")
    print(f"state_keys: {dataset.state_keys}")
    print(f"action_keys: {dataset.action_keys}")

    sample = dataset[0]
    print()
    print("sample[0]:")
    for key, value in sample.items():
        if hasattr(value, "shape"):
            print(f"  {key:<18} shape={tuple(value.shape)} dtype={value.dtype}")
        else:
            print(f"  {key:<18} {value}")


if __name__ == "__main__":
    main()
