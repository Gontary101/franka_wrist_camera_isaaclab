#!/usr/bin/env python3
"""Inspect the USD object catalog."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from franka_wrist_camera_scene.objects.catalog import load_object_catalog


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a USD object catalog.")
    parser.add_argument(
        "--config",
        type=str,
        default="object_catalog.yaml",
        help="Catalog config name under configs/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = load_object_catalog(args.config)

    missing_paths = [
        variant.usd_path
        for category in catalog.categories
        for variant in category.variants
        if not variant.usd_path.exists()
    ]

    print(f"config: {args.config}")
    print(f"asset_root: {catalog.asset_root}")
    print(f"categories: {len(catalog.categories)}")
    print(f"variants: {len(catalog.variants)}")
    print(f"missing files: {len(missing_paths)}")
    print()
    print(f"{'category':<18} {'label':<12} {'split':<8} {'role':<10} {'affordances':<28} {'variants':<8}")

    for category in catalog.categories:
        affordances = ",".join(category.affordances)
        print(
            f"{category.id:<18} "
            f"{category.label:<12} "
            f"{category.split:<8} "
            f"{category.role:<10} "
            f"{affordances:<28} "
            f"{len(category.variants):<8}"
        )

    if missing_paths:
        print()
        print("missing USD files:")
        for path in missing_paths:
            print(f"  {path}")
        raise FileNotFoundError(f"{len(missing_paths)} catalog USD files are missing.")


if __name__ == "__main__":
    main()
