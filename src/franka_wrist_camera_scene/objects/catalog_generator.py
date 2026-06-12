"""Generate USD object catalog configs from an asset directory tree."""

from __future__ import annotations

from pathlib import Path

import yaml

from franka_wrist_camera_scene.utils.paths import REPO_ROOT


SUPPORT_CATEGORIES = {"plate", "tray", "placemat"}
HOLLOW_CATEGORIES = {"bowl", "cup"}
CENTER_TOP_GRASP_CATEGORIES = {
    "apple",
    "avocado",
    "beer",
    "bottle",
    "box",
    "can",
    "egg",
    "kiwi",
    "lemon",
    "lime",
    "onion",
    "orange",
    "peach",
    "potato",
    "tangerine",
    "tomato",
}
VARIANT_AFFORDANCE_OVERRIDES = {
    ("box", "box00"): ["reachable", "container"],
}
VARIANT_GRASP_STRATEGY_OVERRIDES = {
    ("box", "box00"): "unsupported",
}
IGNORED_DIRECTORY_NAMES = {"texture"}


def label_from_variant_stem(stem: str) -> str:
    """Infer a human object label from a USD filename stem."""
    if stem.startswith("dbottle") or stem.startswith("wbottle"):
        return "bottle"
    if stem.startswith("fcan"):
        return "can"

    label = "".join(char for char in stem if not char.isdigit())
    return label.lower()


def affordances_for_label(label: str) -> list[str]:
    if label in SUPPORT_CATEGORIES:
        return ["reachable", "support"]

    if label in HOLLOW_CATEGORIES:
        return ["reachable", "container"]

    if label in CENTER_TOP_GRASP_CATEGORIES:
        return ["pickable", "reachable"]

    raise ValueError(f"No affordance policy defined for category label: {label}")


def role_for_label(label: str) -> str:
    if label in SUPPORT_CATEGORIES:
        return "clutter"
    return "target"


def grasp_strategy_for_label(label: str) -> str:
    if label in SUPPORT_CATEGORIES or label in HOLLOW_CATEGORIES:
        return "unsupported"

    if label in CENTER_TOP_GRASP_CATEGORIES:
        return "center_top"

    raise ValueError(f"No grasp strategy policy defined for category label: {label}")


def category_entry(
    category_id: str,
    label: str,
    split: str,
    variants: list[dict],
) -> dict:
    """Create one catalog category entry."""
    return {
        "id": category_id,
        "label": label,
        "split": split,
        "role": role_for_label(label),
        "affordances": affordances_for_label(label),
        "grasp_strategy": grasp_strategy_for_label(label),
        "variants": variants,
    }


def collect_category_variants(asset_root: Path, category_dir: Path) -> list[dict]:
    """Collect direct USD variants from one category directory."""
    variants: list[dict] = []

    for usd_path in sorted(category_dir.glob("*.usd")):
        variant = {
            "id": usd_path.stem,
            "usd_path": str(usd_path.relative_to(asset_root)),
        }

        override = VARIANT_AFFORDANCE_OVERRIDES.get((category_dir.name, usd_path.stem))
        if override is not None:
            variant["affordances"] = override

        strategy = VARIANT_GRASP_STRATEGY_OVERRIDES.get((category_dir.name, usd_path.stem))
        if strategy is not None:
            variant["grasp_strategy"] = strategy

        variants.append(variant)

    return variants


def collect_unseen_categories(asset_root: Path, unseen_dir: Path) -> list[dict]:
    """Collect unseen USD variants, grouped by inferred object label."""
    grouped_variants: dict[str, list[dict]] = {}

    for usd_path in sorted(unseen_dir.glob("*.usd")):
        label = label_from_variant_stem(usd_path.stem)
        grouped_variants.setdefault(label, []).append(
            {
                "id": usd_path.stem,
                "usd_path": str(usd_path.relative_to(asset_root)),
            }
        )

    return [
        category_entry(
            category_id=f"unseen_{label}",
            label=label,
            split="unseen",
            variants=variants,
        )
        for label, variants in sorted(grouped_variants.items())
    ]


def catalog_asset_root_value(asset_root: Path) -> str:
    return asset_root.relative_to(REPO_ROOT).as_posix()


def generate_object_catalog(asset_root: Path) -> dict:
    """Generate an object catalog dictionary from an asset tree."""
    categories: list[dict] = []

    for category_dir in sorted(path for path in asset_root.iterdir() if path.is_dir()):
        if category_dir.name in IGNORED_DIRECTORY_NAMES:
            continue

        if category_dir.name == "unseen":
            categories.extend(collect_unseen_categories(asset_root, category_dir))
            continue

        variants = collect_category_variants(asset_root, category_dir)
        if not variants:
            continue

        label = category_dir.name
        categories.append(
            category_entry(
                category_id=category_dir.name,
                label=label,
                split="train",
                variants=variants,
            )
        )

    return {
        "asset_root": catalog_asset_root_value(asset_root),
        "categories": categories,
    }


def write_generated_object_catalog(
    asset_root: Path,
    output_path: Path,
) -> Path:
    """Generate and write an object catalog YAML file."""
    catalog = generate_object_catalog(asset_root)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(catalog, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    return output_path
