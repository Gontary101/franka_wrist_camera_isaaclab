#!/usr/bin/env python3
"""Write deterministic train/val splits for an exported ILA dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from franka_wrist_camera_scene.export.ila_splits import write_deterministic_ila_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write deterministic ILA train/val splits.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_path, val_path = write_deterministic_ila_splits(
        dataset_dir=args.dataset_dir,
        val_fraction=args.val_fraction,
    )
    print(f"[INFO] Saved train split to: {train_path}", flush=True)
    print(f"[INFO] Saved val split to: {val_path}", flush=True)


if __name__ == "__main__":
    main()
