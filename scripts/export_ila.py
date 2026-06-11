#!/usr/bin/env python3
"""Export raw tabletop episodes to an image-language-action dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from franka_wrist_camera_scene.export.ila import export_collection_to_ila


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export raw tabletop collection to ILA format.")
    parser.add_argument("raw_collection_dir", type=Path)
    parser.add_argument("export_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = export_collection_to_ila(
        raw_collection_dir=args.raw_collection_dir,
        export_dir=args.export_dir,
    )
    print(f"[INFO] Saved ILA manifest to: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
