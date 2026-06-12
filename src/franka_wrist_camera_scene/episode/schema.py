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
    record_cameras: bool = False
    record_depth: bool = False
    num_camera_frames: int = 0
    object_pos_local: tuple[float, float, float] | None = None
    place_pos_local: tuple[float, float, float] | None = None
    object_xy_offset: tuple[float, float] | None = None
    place_xy_offset: tuple[float, float] | None = None
    object_category_id: str | None = None
    object_variant_id: str | None = None
    object_label: str | None = None
    object_usd_path: str | None = None
    object_yaw_relevant: bool | None = None
    object_planar_aspect_ratio: float | None = None
    object_planar_minor_axis_local: tuple[float, float] | None = None
    object_planar_major_axis_local: tuple[float, float] | None = None
    grasp_closing_axis_xy: tuple[float, float] | None = None
    light_intensity: float | None = None
    light_color: tuple[float, float, float] | None = None

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
