"""Deterministic task-parameter sampling."""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True, slots=True)
class XYRange:
    x: tuple[float, float]
    y: tuple[float, float]


@dataclass(frozen=True, slots=True)
class PickPlaceSample:
    object_xy_offset: tuple[float, float]
    place_xy_offset: tuple[float, float]
    object_color: tuple[float, float, float]


def parse_xy_range(config: dict) -> XYRange:
    return XYRange(
        x=(float(config["x"][0]), float(config["x"][1])),
        y=(float(config["y"][0]), float(config["y"][1])),
    )


def sample_pick_place_offsets(
    seed: int,
    episode_id: int,
    object_range: XYRange,
    place_range: XYRange,
    object_colors: tuple[tuple[float, float, float], ...],
) -> PickPlaceSample:
    rng = random.Random(seed + episode_id)

    object_xy_offset = (
        rng.uniform(object_range.x[0], object_range.x[1]),
        rng.uniform(object_range.y[0], object_range.y[1]),
    )
    place_xy_offset = (
        rng.uniform(place_range.x[0], place_range.x[1]),
        rng.uniform(place_range.y[0], place_range.y[1]),
    )
    object_color = object_colors[rng.randrange(len(object_colors))]

    return PickPlaceSample(
        object_xy_offset=object_xy_offset,
        place_xy_offset=place_xy_offset,
        object_color=object_color,
    )
