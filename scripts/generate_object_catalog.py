#!/usr/bin/env python3
"""Generate a USD object catalog from the local objects asset tree."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from franka_wrist_camera_scene.objects.catalog_generator import write_generated_object_catalog
from franka_wrist_camera_scene.utils.paths import REPO_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a USD object catalog.")
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=REPO_ROOT / "objects",
        help="Root directory containing object USD asset folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "configs" / "object_catalog.generated.yaml",
        help="Generated catalog YAML path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = write_generated_object_catalog(
        asset_root=args.asset_root.resolve(),
        output_path=args.output.resolve(),
    )
    print(f"[INFO] Saved generated object catalog to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
