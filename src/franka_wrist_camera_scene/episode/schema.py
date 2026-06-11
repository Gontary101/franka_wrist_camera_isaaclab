"""Dataclass schemas for recorded tabletop episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EpisodeMetadata:
    """Metadata saved once per recorded episode."""

    episode_id: int
    task_name: str
    instruction: str
    success: bool
    num_steps: int
    sim_dt: float
    seed: int | None = None

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
