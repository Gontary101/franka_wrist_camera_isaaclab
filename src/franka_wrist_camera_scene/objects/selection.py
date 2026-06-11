"""Deterministic object selection helpers."""

from __future__ import annotations

from franka_wrist_camera_scene.objects.catalog import ObjectCatalog, ObjectCategory, ObjectVariant


def find_variant(
    catalog: ObjectCatalog,
    category_id: str,
    variant_id: str,
) -> tuple[ObjectCategory, ObjectVariant]:
    for category in catalog.categories:
        if category.id != category_id:
            continue

        for variant in category.variants:
            if variant.id == variant_id:
                return category, variant

        raise KeyError(f"Variant '{variant_id}' not found in category '{category_id}'.")

    raise KeyError(f"Category '{category_id}' not found in object catalog.")
