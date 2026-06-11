#!/usr/bin/env python3
"""Visualize one exported image-language-action episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one exported ILA episode.")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("episode_id", type=str)
    parser.add_argument("--output", type=Path, default=Path("ila_episode_preview.png"))
    parser.add_argument("--num_frames", type=int, default=8)
    return parser.parse_args()


def load_episode_entry(dataset_dir: Path, episode_id: str) -> dict:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    normalized_id = int(episode_id)

    for episode in manifest["episodes"]:
        if int(episode["episode_id"]) == normalized_id:
            return episode

    raise KeyError(f"Episode {episode_id} not found in {dataset_dir / 'manifest.json'}")


def main() -> None:
    args = parse_args()

    episode_entry = load_episode_entry(args.dataset_dir, args.episode_id)
    episode_path = args.dataset_dir / episode_entry["episode_file"]

    with np.load(episode_path) as episode:
        agent_rgb = episode["agent_rgb"]
        wrist_rgb = episode["wrist_rgb"]
        delta_pos = episode["action_delta_target_pos_w"]
        gripper = episode["action_finger_opening_m"]
        timestamps = episode["timestamps_s"]

        frame_count = int(agent_rgb.shape[0])
        frame_indices = np.linspace(0, frame_count - 1, min(args.num_frames, frame_count), dtype=int)

        action_norm = np.linalg.norm(delta_pos.reshape(frame_count, -1)[:, :3], axis=1)

        fig = plt.figure(figsize=(2.4 * len(frame_indices), 7.0))
        grid = fig.add_gridspec(4, len(frame_indices))

        for col, frame_idx in enumerate(frame_indices):
            ax = fig.add_subplot(grid[0, col])
            ax.imshow(agent_rgb[frame_idx])
            ax.set_title(f"t={timestamps[frame_idx]:.2f}s")
            ax.axis("off")

            ax = fig.add_subplot(grid[1, col])
            ax.imshow(wrist_rgb[frame_idx])
            ax.axis("off")

        action_ax = fig.add_subplot(grid[2, :])
        action_ax.plot(timestamps, action_norm)
        action_ax.set_ylabel("||delta target pos||")
        action_ax.set_xlabel("time [s]")

        gripper_ax = fig.add_subplot(grid[3, :])
        gripper_ax.plot(timestamps, gripper)
        gripper_ax.set_ylabel("gripper opening [m]")
        gripper_ax.set_xlabel("time [s]")

        fig.suptitle(
            f"episode {episode_entry['episode_id']} | "
            f"success={episode_entry['success']} | "
            f"{episode_entry['instruction']}"
        )
        fig.tight_layout()
        fig.savefig(args.output, dpi=140)
        plt.close(fig)

    print(f"[INFO] Saved episode visualization to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
