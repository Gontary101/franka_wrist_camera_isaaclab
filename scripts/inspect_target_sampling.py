#!/usr/bin/env python3
"""Inspect variants eligible for target-object sampling."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from franka_wrist_camera_scene.objects.catalog import load_object_catalog
from franka_wrist_camera_scene.objects.geometry_registry import (
    get_object_geometry,
    load_object_geometry_registry,
)
from franka_wrist_camera_scene.objects.selection import (
    variant_affordances,
    variant_grasp_strategy,
    variant_matches,
)
from franka_wrist_camera_scene.utils.paths import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect target variants eligible for sampling.")
    parser.add_argument("--collection-config", default="collection.yaml")
    parser.add_argument("--only-yaw-relevant", action="store_true")
    parser.add_argument("--category")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_cfg = load_yaml_config(args.collection_config)["target_object"]

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
    required_grasp_strategy = str(target_cfg["required_grasp_strategy"])

    print(f"collection_config: {args.collection_config}")
    print(
        f"{'category':<16} "
        f"{'variant':<16} "
        f"{'affordances':<44} "
        f"{'strategy':<12} "
        f"{'yaw':<6} "
        f"{'aspect':<8} "
        f"usd_path"
    )

    count = 0
    for category in catalog.categories:
        if args.category and category.id != args.category:
            continue
        if category.split != split or category.role != role:
            continue

        for variant in category.variants:
            if not variant_matches(
                category,
                variant,
                required_affordances,
                required_grasp_strategy,
            ):
                continue

            geometry = get_object_geometry(geometry_registry, category.id, variant.id)
            usd_path = variant.usd_path.relative_to(catalog.asset_root).as_posix()
            if geometry.usd_path != usd_path:
                raise ValueError(
                    f"Geometry USD path mismatch for {category.id}/{variant.id}: "
                    f"catalog={usd_path}, geometry={geometry.usd_path}"
                )
            if args.only_yaw_relevant and not geometry.yaw_relevant:
                continue

            count += 1
            print(
                f"{category.id:<16} "
                f"{variant.id:<16} "
                f"{','.join(variant_affordances(category, variant)):<44} "
                f"{variant_grasp_strategy(category, variant):<12} "
                f"{str(geometry.yaw_relevant).lower():<6} "
                f"{geometry.planar_aspect_ratio:<8.3f} "
                f"{usd_path}"
            )

    print(f"\neligible_variants: {count}")


if __name__ == "__main__":
    main()
