from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

from franka_wrist_camera_scene.objects.catalog import load_object_catalog
from franka_wrist_camera_scene.objects.selection import sample_catalog_object


@dataclass(frozen=True, slots=True)
class CatalogObjectContext:
    category_id: str
    variant_id: str
    label: str
    usd_path: Path


def load_catalog_object_context(
    catalog_config: str,
    category_id: str,
    variant_id: str,
    split: str,
    role: str,
    required_affordances: tuple[str, ...],
    rng: random.Random | None = None,
) -> CatalogObjectContext:
    catalog = load_object_catalog(catalog_config)
    category, variant = sample_catalog_object(
        catalog=catalog,
        category_id=category_id,
        variant_id=variant_id,
        split=split,
        role=role,
        required_affordances=required_affordances,
        rng=rng,
    )

    return CatalogObjectContext(
        category_id=category.id,
        variant_id=variant.id,
        label=category.label,
        usd_path=variant.usd_path,
    )


