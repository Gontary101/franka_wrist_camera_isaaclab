"""Planar geometry inference for USD object assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from pxr import Gf, Usd, UsdGeom

from franka_wrist_camera_scene.objects.catalog import ObjectCatalog, load_object_catalog


@dataclass(frozen=True, slots=True)
class PlanarGeometry:
    local_bbox_min: tuple[float, float, float]
    local_bbox_max: tuple[float, float, float]
    local_bbox_size: tuple[float, float, float]
    planar_centroid_local: tuple[float, float]
    planar_major_axis_local: tuple[float, float] | None
    planar_minor_axis_local: tuple[float, float] | None
    planar_extent_major: float
    planar_extent_minor: float
    planar_aspect_ratio: float
    yaw_relevant: bool


@dataclass(frozen=True, slots=True)
class ObjectGeometryRecord:
    category_id: str
    variant_id: str
    usd_path: str
    geometry: PlanarGeometry


def _root_prim(stage: Usd.Stage) -> Usd.Prim:
    default_prim = stage.GetDefaultPrim()
    if default_prim.IsValid():
        return default_prim

    root_prims = tuple(stage.GetPseudoRoot().GetChildren())
    if len(root_prims) != 1:
        raise RuntimeError(
            f"USD stage must have a default prim or exactly one root prim, found {len(root_prims)}."
        )

    return root_prims[0]


def _mesh_points_in_object_frame(stage: Usd.Stage) -> np.ndarray:
    root = _root_prim(stage)
    points: list[tuple[float, float, float]] = []

    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)
        mesh_points = mesh.GetPointsAttr().Get()
        if mesh_points is None or len(mesh_points) == 0:
            continue

        mesh_to_world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )

        for point in mesh_points:
            object_point = mesh_to_world.Transform(Gf.Vec3d(point[0], point[1], point[2]))
            points.append((float(object_point[0]), float(object_point[1]), float(object_point[2])))

    if not points:
        raise RuntimeError(f"No mesh vertices found in USD stage: {stage.GetRootLayer().identifier}")

    return np.asarray(points, dtype=np.float64)


def _canonical_axis(axis: np.ndarray) -> tuple[float, float]:
    axis = axis / np.linalg.norm(axis)

    if abs(axis[0]) >= abs(axis[1]):
        if axis[0] < 0.0:
            axis = -axis
    elif axis[1] < 0.0:
        axis = -axis

    return (float(axis[0]), float(axis[1]))


def infer_planar_geometry_from_usd(
    usd_path: Path,
    aspect_threshold: float,
) -> PlanarGeometry:
    """Infer local planar object geometry from USD mesh vertices."""
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD file: {usd_path}")

    points = _mesh_points_in_object_frame(stage)

    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    bbox_size = max_xyz - min_xyz

    planar = points[:, :2]
    centroid = planar.mean(axis=0)
    centered = planar - centroid

    if centered.shape[0] < 3:
        raise RuntimeError(f"Need at least 3 mesh vertices for planar PCA: {usd_path}")

    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]

    major_axis = eigenvectors[:, order[0]]
    minor_axis = eigenvectors[:, order[1]]

    major_axis = major_axis / np.linalg.norm(major_axis)
    minor_axis = minor_axis / np.linalg.norm(minor_axis)

    major_projection = centered @ major_axis
    minor_projection = centered @ minor_axis

    major_extent = float(major_projection.max() - major_projection.min())
    minor_extent = float(minor_projection.max() - minor_projection.min())

    if major_extent < minor_extent:
        major_axis, minor_axis = minor_axis, major_axis
        major_extent, minor_extent = minor_extent, major_extent

    if minor_extent <= 1e-8:
        raise RuntimeError(f"Degenerate planar geometry for USD file: {usd_path}")

    aspect_ratio = major_extent / minor_extent
    yaw_relevant = bool(aspect_ratio >= aspect_threshold)

    return PlanarGeometry(
        local_bbox_min=tuple(float(value) for value in min_xyz),
        local_bbox_max=tuple(float(value) for value in max_xyz),
        local_bbox_size=tuple(float(value) for value in bbox_size),
        planar_centroid_local=(float(centroid[0]), float(centroid[1])),
        planar_major_axis_local=_canonical_axis(major_axis) if yaw_relevant else None,
        planar_minor_axis_local=_canonical_axis(minor_axis) if yaw_relevant else None,
        planar_extent_major=major_extent,
        planar_extent_minor=minor_extent,
        planar_aspect_ratio=float(aspect_ratio),
        yaw_relevant=yaw_relevant,
    )


def generate_object_geometry_records(
    catalog: ObjectCatalog,
    aspect_threshold: float,
) -> tuple[ObjectGeometryRecord, ...]:
    """Generate planar geometry records for every variant in an object catalog."""
    records: list[ObjectGeometryRecord] = []

    for category in catalog.categories:
        for variant in category.variants:
            geometry = infer_planar_geometry_from_usd(
                usd_path=variant.usd_path,
                aspect_threshold=aspect_threshold,
            )

            records.append(
                ObjectGeometryRecord(
                    category_id=category.id,
                    variant_id=variant.id,
                    usd_path=str(variant.usd_path.relative_to(catalog.asset_root)),
                    geometry=geometry,
                )
            )

    return tuple(records)


def _rounded_list(values: tuple[float, ...], digits: int = 6) -> list[float]:
    return [round(float(value), digits) for value in values]


def _geometry_to_dict(geometry: PlanarGeometry) -> dict:
    return {
        "local_bbox_min": _rounded_list(geometry.local_bbox_min),
        "local_bbox_max": _rounded_list(geometry.local_bbox_max),
        "local_bbox_size": _rounded_list(geometry.local_bbox_size),
        "planar_centroid_local": _rounded_list(geometry.planar_centroid_local),
        "planar_major_axis_local": (
            _rounded_list(geometry.planar_major_axis_local)
            if geometry.planar_major_axis_local is not None
            else None
        ),
        "planar_minor_axis_local": (
            _rounded_list(geometry.planar_minor_axis_local)
            if geometry.planar_minor_axis_local is not None
            else None
        ),
        "planar_extent_major": round(geometry.planar_extent_major, 6),
        "planar_extent_minor": round(geometry.planar_extent_minor, 6),
        "planar_aspect_ratio": round(geometry.planar_aspect_ratio, 6),
        "yaw_relevant": geometry.yaw_relevant,
    }


def write_object_geometry(
    catalog_config: str,
    output_path: Path,
    aspect_threshold: float,
) -> Path:
    """Generate and write planar geometry metadata for a USD object catalog."""
    catalog = load_object_catalog(catalog_config)
    records = generate_object_geometry_records(
        catalog=catalog,
        aspect_threshold=aspect_threshold,
    )

    data = {
        "format_version": 1,
        "catalog_config": catalog_config,
        "aspect_threshold": float(aspect_threshold),
        "records": [
            {
                "category_id": record.category_id,
                "variant_id": record.variant_id,
                "usd_path": record.usd_path,
                **_geometry_to_dict(record.geometry),
            }
            for record in records
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    return output_path
