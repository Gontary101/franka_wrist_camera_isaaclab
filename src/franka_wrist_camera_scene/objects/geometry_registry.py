"""Lookup utilities for generated object geometry metadata."""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from franka_wrist_camera_scene.utils.paths import REPO_ROOT


@dataclass(frozen=True, slots=True)
class ObjectPlanarGeometry:
    """Planar geometry in the authored USD object coordinate frame."""

    usd_path: str
    local_bbox_size: tuple[float, float, float]
    planar_centroid_local: tuple[float, float]
    planar_major_axis_local: tuple[float, float] | None
    planar_minor_axis_local: tuple[float, float] | None
    planar_extent_major: float
    planar_extent_minor: float
    planar_aspect_ratio: float
    yaw_relevant: bool


@dataclass(frozen=True, slots=True)
class ObjectGeometryRegistry:
    catalog_config: str
    records: dict[tuple[str, str], ObjectPlanarGeometry]


def _tuple3(values: list[float]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _tuple2(values: list[float]) -> tuple[float, float]:
    return (float(values[0]), float(values[1]))


def _tuple2_or_none(values: list[float] | None) -> tuple[float, float] | None:
    if values is None:
        return None
    return _tuple2(values)


def load_object_geometry_registry(
    geometry_config: str = "object_geometry.generated.yaml",
) -> ObjectGeometryRegistry:
    path = REPO_ROOT / "configs" / geometry_config
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    if data["format_version"] != 1:
        raise ValueError(f"Unsupported object geometry format_version: {data['format_version']}")

    registry: dict[tuple[str, str], ObjectPlanarGeometry] = {}

    for record in data["records"]:
        key = (str(record["category_id"]), str(record["variant_id"]))
        if key in registry:
            raise ValueError(f"Duplicate object geometry record: {key}")

        registry[key] = ObjectPlanarGeometry(
            usd_path=str(record["usd_path"]),
            local_bbox_size=_tuple3(record["local_bbox_size"]),
            planar_centroid_local=_tuple2(record["planar_centroid_local"]),
            planar_major_axis_local=_tuple2_or_none(record["planar_major_axis_local"]),
            planar_minor_axis_local=_tuple2_or_none(record["planar_minor_axis_local"]),
            planar_extent_major=float(record["planar_extent_major"]),
            planar_extent_minor=float(record["planar_extent_minor"]),
            planar_aspect_ratio=float(record["planar_aspect_ratio"]),
            yaw_relevant=bool(record["yaw_relevant"]),
        )

    return ObjectGeometryRegistry(
        catalog_config=str(data["catalog_config"]),
        records=registry,
    )


def get_object_geometry(
    registry: ObjectGeometryRegistry,
    category_id: str,
    variant_id: str,
) -> ObjectPlanarGeometry:
    key = (category_id, variant_id)
    if key not in registry.records:
        raise KeyError(f"Missing object geometry record for {category_id}/{variant_id}")
    return registry.records[key]
