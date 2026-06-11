"""Scene lighting utilities."""

from __future__ import annotations

from pxr import Gf, UsdLux
from isaaclab.scene import InteractiveScene


def set_dome_light(scene: InteractiveScene, intensity: float, color_rgb: tuple[float, float, float]) -> None:
    """Set the dome light intensity and color in the USD stage."""
    light_path = "/World/Light"
    light_prim = scene.stage.GetPrimAtPath(light_path)
    if not light_prim.IsValid():
        raise RuntimeError(f"Dome light prim not found: {light_path}")

    light = UsdLux.DomeLight(light_prim)
    light.GetIntensityAttr().Set(float(intensity))
    light.GetColorAttr().Set(Gf.Vec3f(*color_rgb))
