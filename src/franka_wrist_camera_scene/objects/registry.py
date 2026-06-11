"""Object registry for tabletop manipulation assets."""

from __future__ import annotations

from dataclasses import dataclass

from franka_wrist_camera_scene.utils.paths import load_yaml_config


@dataclass(frozen=True, slots=True)
class ObjectColor:
    name: str
    rgb: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class ObjectGraspSpec:
    tcp_offset_local: tuple[float, float, float]
    pregrasp_height_m: float
    lift_height_m: float


@dataclass(frozen=True, slots=True)
class ObjectSpec:
    id: str
    label: str
    category: str
    kind: str
    size: tuple[float, float, float]
    default_color: ObjectColor
    grasp: ObjectGraspSpec
    aliases: tuple[str, ...]


def load_object_registry(config_name: str = "objects.yaml") -> dict[str, ObjectSpec]:
    data = load_yaml_config(config_name)
    objects: dict[str, ObjectSpec] = {}

    for item in data["objects"]:
        object_id = str(item["id"])
        color = item["default_color"]
        grasp = item["grasp"]
        language = item["language"]

        objects[object_id] = ObjectSpec(
            id=object_id,
            label=str(item["label"]),
            category=str(item["category"]),
            kind=str(item["kind"]),
            size=tuple(float(x) for x in item["size"]),
            default_color=ObjectColor(
                name=str(color["name"]),
                rgb=tuple(float(x) for x in color["rgb"]),
            ),
            grasp=ObjectGraspSpec(
                tcp_offset_local=tuple(float(x) for x in grasp["tcp_offset_local"]),
                pregrasp_height_m=float(grasp["pregrasp_height_m"]),
                lift_height_m=float(grasp["lift_height_m"]),
            ),
            aliases=tuple(str(x) for x in language["aliases"]),
        )

    return objects
