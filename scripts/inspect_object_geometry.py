#!/usr/bin/env python3
"""Inspect generated USD object planar geometry metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from franka_wrist_camera_scene.utils.paths import REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect generated object geometry metadata.")
    parser.add_argument(
        "--geometry-config",
        type=Path,
        default=REPO_ROOT / "configs" / "object_geometry.generated.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = yaml.safe_load(args.geometry_config.read_text(encoding="utf-8"))
    records = data["records"]

    yaw_relevant = [record for record in records if record["yaw_relevant"]]

    print(f"geometry_config: {args.geometry_config}")
    print(f"catalog_config: {data['catalog_config']}")
    print(f"records: {len(records)}")
    print(f"yaw_relevant: {len(yaw_relevant)}")
    print(f"yaw_irrelevant: {len(records) - len(yaw_relevant)}")
    print()
    print(
        f"{'category':<16} "
        f"{'variant':<16} "
        f"{'aspect':<10} "
        f"{'yaw':<6} "
        f"{'major_axis':<24} "
        f"{'minor_axis':<24}"
    )

    for record in records:
        if not record["yaw_relevant"]:
            continue

        print(
            f"{record['category_id']:<16} "
            f"{record['variant_id']:<16} "
            f"{record['planar_aspect_ratio']:<10.3f} "
            f"{str(record['yaw_relevant']):<6} "
            f"{str(record['planar_major_axis_local']):<24} "
            f"{str(record['planar_minor_axis_local']):<24}"
        )


if __name__ == "__main__":
    main()
