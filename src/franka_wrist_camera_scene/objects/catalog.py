"""USD object catalog for tabletop manipulation assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from franka_wrist_camera_scene.utils.paths import REPO_ROOT, load_yaml_config


@dataclass(frozen=True, slots=True)
class ObjectVariant:
    """One concrete USD asset variant."""

    id: str
    usd_path: Path
    affordances: tuple[str, ...] | None = None
    grasp_strategy: str | None = None


@dataclass(frozen=True, slots=True)
class ObjectCategory:
    """Object category containing one or more USD variants."""

    id: str
    label: str
    split: str
    role: str
    affordances: tuple[str, ...]
    grasp_strategy: str
    variants: tuple[ObjectVariant, ...]


@dataclass(frozen=True, slots=True)
class ObjectCatalog:
    """Loaded USD object catalog."""

    asset_root: Path
    categories: tuple[ObjectCategory, ...]

    @property
    def variants(self) -> tuple[ObjectVariant, ...]:
        return tuple(variant for category in self.categories for variant in category.variants)


def resolve_asset_root(asset_root_value: str) -> Path:
    asset_root = Path(asset_root_value)
    if asset_root.is_absolute():
        raise ValueError(f"Object catalog asset_root must be repo-relative, got: {asset_root}")
    return REPO_ROOT / asset_root


def load_object_catalog(config_name: str = "object_catalog.yaml") -> ObjectCatalog:
    """Load the USD object catalog from configs/."""
    data = load_yaml_config(config_name)
    asset_root = resolve_asset_root(str(data["asset_root"]))

    categories: list[ObjectCategory] = []
    for item in data["categories"]:
        variants = tuple(
            ObjectVariant(
                id=str(variant["id"]),
                usd_path=asset_root / str(variant["usd_path"]),
                affordances=(
                    tuple(str(value) for value in variant["affordances"])
                    if "affordances" in variant
                    else None
                ),
                grasp_strategy=(
                    str(variant["grasp_strategy"])
                    if "grasp_strategy" in variant
                    else None
                ),
            )
            for variant in item["variants"]
        )

        categories.append(
            ObjectCategory(
                id=str(item["id"]),
                label=str(item["label"]),
                split=str(item["split"]),
                role=str(item["role"]),
                affordances=tuple(str(value) for value in item["affordances"]),
                grasp_strategy=str(item["grasp_strategy"]),
                variants=variants,
            )
        )

    return ObjectCatalog(
        asset_root=asset_root,
        categories=tuple(categories),
    )
