#!/usr/bin/env python3
"""Inspect registered manipulation objects."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(REPO_SRC))

from franka_wrist_camera_scene.objects.registry import load_object_registry


def main() -> None:
    objects = load_object_registry()

    print(f"objects: {len(objects)}")
    print(f"{'id':<22} {'label':<14} {'category':<14} {'kind':<10} {'size'}")

    for spec in objects.values():
        print(f"{spec.id:<22} {spec.label:<14} {spec.category:<14} {spec.kind:<10} {spec.size}")


if __name__ == "__main__":
    main()
