"""Deterministic split writer for exported ILA datasets."""

from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_split_file(dataset_dir: Path, split_name: str, episodes: list[dict]) -> Path:
    split = {
        "split": split_name,
        "dataset_type": "image_language_action",
        "episode_ids": [int(episode["episode_id"]) for episode in episodes],
        "episode_files": [episode["episode_file"] for episode in episodes],
    }

    split_dir = dataset_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    split_path = split_dir / f"{split_name}.json"
    split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    return split_path


def write_deterministic_ila_splits(
    dataset_dir: Path,
    val_fraction: float,
) -> tuple[Path, Path]:
    manifest = load_json(dataset_dir / "manifest.json")
    episodes = sorted(manifest["episodes"], key=lambda item: int(item["episode_id"]))

    num_episodes = len(episodes)
    num_val = max(1, round(num_episodes * val_fraction))
    num_val = min(num_val, num_episodes - 1)

    train_episodes = episodes[:-num_val]
    val_episodes = episodes[-num_val:]

    train_path = write_split_file(dataset_dir, "train", train_episodes)
    val_path = write_split_file(dataset_dir, "val", val_episodes)

    return train_path, val_path
