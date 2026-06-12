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


def sample_catalog_object(
    catalog: ObjectCatalog,
    category_id: str | None = None,
    variant_id: str | None = None,
    split: str | None = None,
    rng: random.Random | None = None,
) -> tuple[ObjectCategory, ObjectVariant]:
    """Sample a category and variant from the catalog based on filters.

    If category_id or variant_id is "sample" or None, they are sampled.
    If split is specified, only categories matching that split are sampled.
    """
    import random

    if rng is None:
        rng = random.Random()

    # If specific category is requested
    if category_id is not None and category_id != "sample":
        target_category = None
        for category in catalog.categories:
            if category.id == category_id:
                target_category = category
                break
        if target_category is None:
            raise KeyError(f"Category '{category_id}' not found in object catalog.")

        # If specific variant is requested
        if variant_id is not None and variant_id != "sample":
            target_variant = None
            for variant in target_category.variants:
                if variant.id == variant_id:
                    target_variant = variant
                    break
            if target_variant is None:
                raise KeyError(f"Variant '{variant_id}' not found in category '{category_id}'.")
            return target_category, target_variant
        else:
            if not target_category.variants:
                raise ValueError(f"Category '{category_id}' has no variants to sample from.")
            sampled_variant = rng.choice(target_category.variants)
            return target_category, sampled_variant

    # Otherwise, sample category first
    categories = catalog.categories
    if split is not None and split != "all":
        categories = tuple(c for c in categories if c.split == split)

    if not categories:
        raise ValueError(f"No categories found in catalog matching split '{split}'.")

    sampled_category = rng.choice(categories)
    if not sampled_category.variants:
        raise ValueError(f"Sampled category '{sampled_category.id}' has no variants.")

    sampled_variant = rng.choice(sampled_category.variants)
    return sampled_category, sampled_variant

