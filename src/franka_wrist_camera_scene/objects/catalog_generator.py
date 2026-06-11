"""Generate USD object catalog configs from an asset directory tree."""

from __future__ import annotations

from pathlib import Path

import yaml

from franka_wrist_camera_scene.utils.paths import REPO_ROOT


SUPPORT_CATEGORIES = {"plate", "tray", "placemat"}
IGNORED_DIRECTORY_NAMES = {"texture"}


def label_from_variant_stem(stem: str) -> str:
    """Infer a human object label from a USD filename stem."""
    if stem.startswith("dbottle") or stem.startswith("wbottle"):
        return "bottle"
    if stem.startswith("fcan"):
        return "can"

    label = "".join(char for char in stem if not char.isdigit())
    return label.lower()


def category_entry(
    category_id: str,
    label: str,
    split: str,
    variants: list[dict],
) -> dict:
    """Create one catalog category entry."""
    is_support = label in SUPPORT_CATEGORIES

    return {
        "id": category_id,
        "label": label,
        "split": split,
        "role": "clutter" if is_support else "target",
        "affordances": ["reachable", "support"] if is_support else ["pickable", "reachable"],
        "variants": variants,
    }


def collect_category_variants(asset_root: Path, category_dir: Path) -> list[dict]:
    """Collect direct USD variants from one category directory."""
    variants: list[dict] = []

    for usd_path in sorted(category_dir.glob("*.usd")):
        variants.append(
            {
                "id": usd_path.stem,
                "usd_path": str(usd_path.relative_to(asset_root)),
            }
        )

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
        "asset_root": str(asset_root.relative_to(REPO_ROOT)),
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
