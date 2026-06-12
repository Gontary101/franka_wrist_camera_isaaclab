#!/usr/bin/env python3
"""Inspect variants eligible for target-object sampling."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from franka_wrist_camera_scene.objects.catalog import (
    ObjectCategory,
    ObjectVariant,
    load_object_catalog,
)
from franka_wrist_camera_scene.objects.geometry_registry import (
    ObjectPlanarGeometry,
    get_object_geometry,
    load_object_geometry_registry,
)
from franka_wrist_camera_scene.objects.selection import variant_affordances, variant_matches
from franka_wrist_camera_scene.utils.paths import REPO_ROOT, load_yaml_config


@dataclass(frozen=True, slots=True)
class SamplingTarget:
    category: ObjectCategory
    variant: ObjectVariant
    affordances: tuple[str, ...]
    geometry: ObjectPlanarGeometry
    usd_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect target variants eligible for sampling.")
    parser.add_argument(
        "--collection-config",
        type=str,
        default="collection.yaml",
        help="Collection config name under configs/.",
    )
    parser.add_argument(
        "--only-yaw-relevant",
        action="store_true",
        help="Show only variants whose generated planar geometry marks yaw as relevant.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Show only one catalog category id.",
    )
    return parser.parse_args()


def catalog_relative_usd_path(catalog_asset_root: Path, variant: ObjectVariant) -> str:
    return variant.usd_path.relative_to(catalog_asset_root).as_posix()


def load_sampling_targets(
    collection_config: str,
    category_filter: str | None,
    only_yaw_relevant: bool,
) -> tuple[SamplingTarget, ...]:
    collection_cfg = load_yaml_config(collection_config)
    target_cfg = collection_cfg["target_object"]

    catalog_config = str(target_cfg["catalog_config"])
    geometry_config = str(target_cfg["geometry_config"])
    catalog = load_object_catalog(catalog_config)
    geometry_registry = load_object_geometry_registry(geometry_config)

    if geometry_registry.catalog_config != catalog_config:
        raise ValueError(
            f"Geometry config {geometry_config} was generated for "
            f"{geometry_registry.catalog_config}, not {catalog_config}."
        )

    split = str(target_cfg["split"])
    role = str(target_cfg["role"])
    required_affordances = tuple(str(value) for value in target_cfg["required_affordances"])

    targets: list[SamplingTarget] = []
    for category in catalog.categories:
        if category_filter is not None and category.id != category_filter:
            continue
        if category.split != split or category.role != role:
            continue

        for variant in category.variants:
            if not variant_matches(category, variant, required_affordances):
                continue

            geometry = get_object_geometry(
                registry=geometry_registry,
                category_id=category.id,
                variant_id=variant.id,
            )
            usd_path = catalog_relative_usd_path(catalog.asset_root, variant)
            if geometry.usd_path != usd_path:
                raise ValueError(
                    "Geometry record USD path mismatch for "
                    f"{category.id}/{variant.id}: catalog={usd_path}, geometry={geometry.usd_path}"
                )
            if only_yaw_relevant and not geometry.yaw_relevant:
                continue

            targets.append(
                SamplingTarget(
                    category=category,
                    variant=variant,
                    affordances=variant_affordances(category, variant),
                    geometry=geometry,
                    usd_path=usd_path,
                )
            )

    return tuple(targets)


def print_targets(targets: tuple[SamplingTarget, ...]) -> None:
    print(
        f"{'category':<16} "
        f"{'variant':<16} "
        f"{'affordances':<44} "
        f"{'yaw':<6} "
        f"{'aspect':<8} "
        f"usd_path"
    )

    for target in targets:
        affordances = ",".join(target.affordances)
        print(
            f"{target.category.id:<16} "
            f"{target.variant.id:<16} "
            f"{affordances:<44} "
            f"{str(target.geometry.yaw_relevant).lower():<6} "
            f"{target.geometry.planar_aspect_ratio:<8.3f} "
            f"{target.usd_path}"
        )


def main() -> None:
    args = parse_args()
    targets = load_sampling_targets(
        collection_config=args.collection_config,
        category_filter=args.category,
        only_yaw_relevant=bool(args.only_yaw_relevant),
    )

    print(f"collection_config: {REPO_ROOT / 'configs' / args.collection_config}")
    print(f"eligible_variants: {len(targets)}")
    print()
    print_targets(targets)


if __name__ == "__main__":
    main()
