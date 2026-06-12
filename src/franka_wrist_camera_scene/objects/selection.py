"""Deterministic object selection helpers."""

from __future__ import annotations

import random

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


def category_matches(
    category: ObjectCategory,
    split: str,
    role: str,
    required_affordances: tuple[str, ...],
) -> bool:
    if category.split != split:
        return False
    if category.role != role:
        return False

    affordances = set(category.affordances)
    return all(affordance in affordances for affordance in required_affordances)


def filtered_categories(
    catalog: ObjectCatalog,
    split: str,
    role: str,
    required_affordances: tuple[str, ...],
) -> tuple[ObjectCategory, ...]:
    categories = tuple(
        category
        for category in catalog.categories
        if category_matches(
            category=category,
            split=split,
            role=role,
            required_affordances=required_affordances,
        )
    )

    if not categories:
        raise ValueError(
            "No catalog categories match "
            f"split={split!r}, role={role!r}, required_affordances={required_affordances!r}."
        )

    return categories


def sample_catalog_object(
    catalog: ObjectCatalog,
    category_id: str,
    variant_id: str,
    split: str,
    role: str,
    required_affordances: tuple[str, ...],
    rng: random.Random | None = None,
) -> tuple[ObjectCategory, ObjectVariant]:
    """Resolve or sample one catalog object under explicit target constraints."""
    categories = filtered_categories(
        catalog=catalog,
        split=split,
        role=role,
        required_affordances=required_affordances,
    )

    should_sample_category = category_id == "sample"
    should_sample_variant = variant_id == "sample"

    if should_sample_category or should_sample_variant:
        if rng is None:
            raise ValueError("A seeded RNG is required when sampling catalog objects.")

    if should_sample_category:
        if not should_sample_variant:
            raise ValueError("variant_id must be 'sample' when category_id is 'sample'.")

        category = rng.choice(categories)
        if not category.variants:
            raise ValueError(f"Sampled category '{category.id}' has no variants.")
        return category, rng.choice(category.variants)

    matching_categories = tuple(category for category in categories if category.id == category_id)
    if not matching_categories:
        raise KeyError(
            f"Category '{category_id}' does not match "
            f"split={split!r}, role={role!r}, required_affordances={required_affordances!r}."
        )

    category = matching_categories[0]

    if should_sample_variant:
        if not category.variants:
            raise ValueError(f"Category '{category.id}' has no variants.")
        return category, rng.choice(category.variants)

    for variant in category.variants:
        if variant.id == variant_id:
            return category, variant

    raise KeyError(f"Variant '{variant_id}' not found in category '{category_id}'.")


