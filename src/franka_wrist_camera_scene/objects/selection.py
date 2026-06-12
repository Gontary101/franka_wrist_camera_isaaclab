"""Deterministic object selection helpers."""

from __future__ import annotations

import random

from franka_wrist_camera_scene.objects.catalog import ObjectCatalog, ObjectCategory, ObjectVariant


def variant_affordances(category: ObjectCategory, variant: ObjectVariant) -> tuple[str, ...]:
    if variant.affordances is not None:
        return variant.affordances
    return category.affordances


def variant_grasp_strategy(category: ObjectCategory, variant: ObjectVariant) -> str:
    if variant.grasp_strategy is not None:
        return variant.grasp_strategy
    return category.grasp_strategy


def variant_matches(
    category: ObjectCategory,
    variant: ObjectVariant,
    required_affordances: tuple[str, ...],
    required_grasp_strategy: str,
) -> bool:
    affordances = set(variant_affordances(category, variant))
    return (
        variant_grasp_strategy(category, variant) == required_grasp_strategy
        and all(affordance in affordances for affordance in required_affordances)
    )


def matching_variants(
    category: ObjectCategory,
    required_affordances: tuple[str, ...],
    required_grasp_strategy: str,
) -> tuple[ObjectVariant, ...]:
    return tuple(
        variant
        for variant in category.variants
        if variant_matches(
            category=category,
            variant=variant,
            required_affordances=required_affordances,
            required_grasp_strategy=required_grasp_strategy,
        )
    )


def filtered_categories(
    catalog: ObjectCatalog,
    split: str,
    role: str,
) -> tuple[ObjectCategory, ...]:
    categories = tuple(
        category
        for category in catalog.categories
        if category.split == split and category.role == role
    )

    if not categories:
        raise ValueError(f"No catalog categories match split={split!r}, role={role!r}.")

    return categories


def sample_catalog_object(
    catalog: ObjectCatalog,
    category_id: str,
    variant_id: str,
    split: str,
    role: str,
    required_affordances: tuple[str, ...],
    required_grasp_strategy: str,
    rng: random.Random | None = None,
) -> tuple[ObjectCategory, ObjectVariant]:
    """Resolve or sample one catalog object under explicit target constraints."""
    categories = filtered_categories(
        catalog=catalog,
        split=split,
        role=role,
    )

    should_sample_category = category_id == "sample"
    should_sample_variant = variant_id == "sample"

    if should_sample_category or should_sample_variant:
        if rng is None:
            raise ValueError("A seeded RNG is required when sampling catalog objects.")

    if should_sample_category:
        if not should_sample_variant:
            raise ValueError("variant_id must be 'sample' when category_id is 'sample'.")

        candidate_categories = tuple(
            (category, variants)
            for category in categories
            if (
                variants := matching_variants(
                    category,
                    required_affordances,
                    required_grasp_strategy,
                )
            )
        )
        if not candidate_categories:
            raise ValueError(
                "No catalog variants match "
                f"required_affordances={required_affordances!r}, "
                f"required_grasp_strategy={required_grasp_strategy!r}."
            )

        category, variants = rng.choice(candidate_categories)
        return category, rng.choice(variants)

    matching_categories = tuple(category for category in categories if category.id == category_id)
    if not matching_categories:
        raise KeyError(
            f"Category '{category_id}' does not match split={split!r}, role={role!r}."
        )

    category = matching_categories[0]

    if should_sample_variant:
        variants = matching_variants(
            category,
            required_affordances,
            required_grasp_strategy,
        )
        if not variants:
            raise ValueError(
                f"Category '{category.id}' has no variants matching "
                f"required_affordances={required_affordances!r}, "
                f"required_grasp_strategy={required_grasp_strategy!r}."
            )
        return category, rng.choice(variants)

    for variant in category.variants:
        if variant.id == variant_id:
            if not variant_matches(
                category,
                variant,
                required_affordances,
                required_grasp_strategy,
            ):
                raise ValueError(
                    f"Variant '{category.id}/{variant.id}' does not match "
                    f"required_affordances={required_affordances!r}; "
                    f"required_grasp_strategy={required_grasp_strategy!r}; "
                    f"variant_affordances={variant_affordances(category, variant)!r}; "
                    f"variant_grasp_strategy={variant_grasp_strategy(category, variant)!r}."
                )
            return category, variant

    raise KeyError(f"Variant '{variant_id}' not found in category '{category_id}'.")
