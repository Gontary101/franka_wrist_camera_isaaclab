"""Statistics writer for exported image-language-action datasets."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_parts(episode: np.lib.npyio.NpzFile, keys: list[str]) -> np.ndarray:
    parts = [np.asarray(episode[key]).reshape(episode[key].shape[0], -1) for key in keys]
    return np.concatenate(parts, axis=1)


def vector_stats(values: np.ndarray) -> dict:
    values = values.astype(np.float64, copy=False)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
    }


def write_ila_dataset_stats(dataset_dir: Path) -> Path:
    manifest_path = dataset_dir / "manifest.json"
    manifest = load_json(manifest_path)

    action_keys = list(manifest["action_keys"])
    state_keys = list(manifest["state_keys"])

    action_batches: list[np.ndarray] = []
    state_batches: list[np.ndarray] = []

    for episode_entry in manifest["episodes"]:
        episode_path = dataset_dir / episode_entry["episode_file"]
        with np.load(episode_path) as episode:
            action_batches.append(flatten_parts(episode, action_keys))
            state_batches.append(flatten_parts(episode, state_keys))

    actions = np.concatenate(action_batches, axis=0)
    states = np.concatenate(state_batches, axis=0)

    stats = {
        "format_version": 1,
        "dataset_type": manifest["dataset_type"],
        "num_frames": int(actions.shape[0]),
        "action_keys": action_keys,
        "state_keys": state_keys,
        "action": vector_stats(actions),
        "state": vector_stats(states),
    }

    stats_path = dataset_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats_path
