#!/usr/bin/env python3
"""Generate planar geometry metadata for USD catalog objects."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from franka_wrist_camera_scene.objects.geometry import write_object_geometry
from franka_wrist_camera_scene.utils.paths import REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate USD object planar geometry metadata.")
    parser.add_argument(
        "--catalog-config",
        type=str,
        default="object_catalog.generated.yaml",
        help="Catalog config name under configs/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "configs" / "object_geometry.generated.yaml",
        help="Output geometry YAML path.",
    )
    parser.add_argument(
        "--aspect-threshold",
        type=float,
        default=1.25,
        help="Planar aspect ratio above which object yaw is considered relevant.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = write_object_geometry(
        catalog_config=args.catalog_config,
        output_path=args.output,
        aspect_threshold=args.aspect_threshold,
    )
    print(f"[INFO] Saved object geometry metadata to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
