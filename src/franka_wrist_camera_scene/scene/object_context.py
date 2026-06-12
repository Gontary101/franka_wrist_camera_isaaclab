from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from franka_wrist_camera_scene.objects.catalog import ObjectCatalog, ObjectVariant, load_object_catalog
from franka_wrist_camera_scene.objects.geometry_registry import (
    ObjectGeometryRegistry,
    ObjectPlanarGeometry,
    get_object_geometry,
    load_object_geometry_registry,
)
from franka_wrist_camera_scene.objects.selection import sample_catalog_object


@dataclass(frozen=True, slots=True)
class CatalogObjectContext:
    category_id: str
    variant_id: str
    label: str
    usd_path: Path
    geometry: ObjectPlanarGeometry


def _validate_geometry_catalog_config(
    registry: ObjectGeometryRegistry,
    geometry_config: str,
    catalog_config: str,
) -> None:
    if registry.catalog_config != catalog_config:
        raise ValueError(
            f"Geometry config {geometry_config} was generated for "
            f"{registry.catalog_config}, not {catalog_config}."
        )


def _validate_geometry_usd_path(
    catalog: ObjectCatalog,
    variant: ObjectVariant,
    geometry: ObjectPlanarGeometry,
    category_id: str,
) -> None:
    catalog_relative_usd_path = variant.usd_path.relative_to(catalog.asset_root).as_posix()
    if geometry.usd_path != catalog_relative_usd_path:
        raise ValueError(
            "Geometry record USD path mismatch for "
            f"{category_id}/{variant.id}: catalog={catalog_relative_usd_path}, "
            f"geometry={geometry.usd_path}"
        )


def load_catalog_object_context(
    catalog_config: str,
    geometry_config: str,
    category_id: str,
    variant_id: str,
    split: str,
    role: str,
    required_affordances: tuple[str, ...],
    rng: random.Random | None = None,
) -> CatalogObjectContext:
    catalog = load_object_catalog(catalog_config)
    geometry_registry = load_object_geometry_registry(geometry_config)
    _validate_geometry_catalog_config(
        registry=geometry_registry,
        geometry_config=geometry_config,
        catalog_config=catalog_config,
    )
    category, variant = sample_catalog_object(
        catalog=catalog,
        category_id=category_id,
        variant_id=variant_id,
        split=split,
        role=role,
        required_affordances=required_affordances,
        rng=rng,
    )
    geometry = get_object_geometry(
        registry=geometry_registry,
        category_id=category.id,
        variant_id=variant.id,
    )
    _validate_geometry_usd_path(
        catalog=catalog,
        variant=variant,
        geometry=geometry,
        category_id=category.id,
    )

    return CatalogObjectContext(
        category_id=category.id,
        variant_id=variant.id,
        label=category.label,
        usd_path=variant.usd_path,
        geometry=geometry,
    )
