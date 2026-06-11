#!/usr/bin/env python3
"""Collect deterministic pick-place episodes in the tabletop scene."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

# Import launcher to apply Isaac Sim 6.0 and pxr compatibility patches before importing isaaclab
from franka_wrist_camera_scene.app import launcher  # noqa: F401
from isaaclab.app import AppLauncher  # noqa: E402
from franka_wrist_camera_scene.utils.paths import load_yaml_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect deterministic pick-and-place tabletop episodes.")
    parser.add_argument(
        "--collection_config",
        type=str,
        default="collection.yaml",
        help="Collection config file under configs/.",
    )
    # Add app launcher arguments
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.kit_args = f"{args.kit_args} --/rtx/hydra/readTransformsFromFabricInRenderDelegate=false".strip()
    return args


def preflight_collection_output(collection_cfg: dict) -> None:
    """Preflight check on output paths before launching simulator."""
    output_dir = Path(collection_cfg["output_dir"])
    start_episode_id = int(collection_cfg["start_episode_id"])
    num_episodes = int(collection_cfg["num_episodes"])

    for episode_id in range(start_episode_id, start_episode_id + num_episodes):
        episode_dir = output_dir / f"{episode_id:06d}"
        if episode_dir.exists():
            raise FileExistsError(f"Episode directory already exists: {episode_dir}")

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Collection manifest already exists: {manifest_path}")


def main() -> None:
    args_cli = parse_args()
    collection_cfg = load_yaml_config(args_cli.collection_config)
    preflight_collection_output(collection_cfg)

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    launcher.patch_physx_schema()

    from franka_wrist_camera_scene.collection.pick_place import collect_pick_place_dataset

    collect_pick_place_dataset(
        collection_cfg=collection_cfg,
        device=args_cli.device,
        simulation_app=simulation_app,
    )

    simulation_app.close()


if __name__ == "__main__":
    main()
