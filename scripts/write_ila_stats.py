#!/usr/bin/env python3
"""Write normalization statistics for an exported ILA dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from franka_wrist_camera_scene.export.ila_stats import write_ila_dataset_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write ILA dataset statistics.")
    parser.add_argument("dataset_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats_path = write_ila_dataset_stats(args.dataset_dir)
    print(f"[INFO] Saved ILA stats to: {stats_path}", flush=True)


if __name__ == "__main__":
    main()
