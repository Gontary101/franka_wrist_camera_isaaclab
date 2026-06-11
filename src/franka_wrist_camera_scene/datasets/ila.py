"""PyTorch dataset for exported image-language-action episodes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ILADataset(Dataset):
    """Frame-level dataset for exported image-language-action episodes."""

    def __init__(self, dataset_dir: Path | str, split: str | None = None):
        self.dataset_dir = Path(dataset_dir)
        self.manifest = json.loads((self.dataset_dir / "manifest.json").read_text(encoding="utf-8"))

        episodes = self.manifest["episodes"]
        if split is not None:
            split_data = json.loads((self.dataset_dir / "splits" / f"{split}.json").read_text(encoding="utf-8"))
            episode_files = set(split_data["episode_files"])
            episodes = [episode for episode in episodes if episode["episode_file"] in episode_files]

        self.episodes = episodes
        self.observation_keys = tuple(self.manifest["observation_keys"])
        self.action_keys = tuple(self.manifest["action_keys"])
        self.state_keys = tuple(self.manifest["state_keys"])

        self.index: list[tuple[int, int]] = []
        self._episode_cache: dict[int, dict[str, np.ndarray]] = {}

        for episode_idx, episode in enumerate(self.episodes):
            episode_path = self.dataset_dir / episode["episode_file"]
            with np.load(episode_path) as data:
                num_frames = int(data["timestamps_s"].shape[0])

            for frame_idx in range(num_frames):
                self.index.append((episode_idx, frame_idx))

    def __len__(self) -> int:
        return len(self.index)

    def _load_episode(self, episode_idx: int) -> dict[str, np.ndarray]:
        if episode_idx not in self._episode_cache:
            episode_path = self.dataset_dir / self.episodes[episode_idx]["episode_file"]
            with np.load(episode_path) as data:
                self._episode_cache[episode_idx] = {key: data[key] for key in data.files}
        return self._episode_cache[episode_idx]

    @staticmethod
    def _rgb_to_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0

    @staticmethod
    def _vector_to_tensor(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(array)).float().reshape(-1)

    def __getitem__(self, item_idx: int) -> dict:
        episode_idx, frame_idx = self.index[item_idx]
        episode_meta = self.episodes[episode_idx]
        episode = self._load_episode(episode_idx)

        sample = {
            "instruction": episode_meta["instruction"],
            "episode_id": int(episode_meta["episode_id"]),
            "frame_index": int(frame_idx),
            "timestamp_s": float(episode["timestamps_s"][frame_idx]),
        }

        for key in self.observation_keys:
            sample[key] = self._rgb_to_tensor(episode[key][frame_idx])

        action_parts = [self._vector_to_tensor(episode[key][frame_idx]) for key in self.action_keys]
        state_parts = [self._vector_to_tensor(episode[key][frame_idx]) for key in self.state_keys]

        sample["action"] = torch.cat(action_parts, dim=0)
        sample["state"] = torch.cat(state_parts, dim=0)

        return sample
