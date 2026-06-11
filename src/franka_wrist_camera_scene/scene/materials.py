"""Scene visual material utilities."""

from __future__ import annotations

import re
from pxr import Gf, UsdShade
from isaaclab.scene import InteractiveScene


def set_target_cube_color(scene: InteractiveScene, color_rgb: tuple[float, float, float]) -> None:
    """Set the target cube visual material color per episode."""
    prim_path_template = scene["target_cube"].cfg.prim_path

    for env_id in range(scene.num_envs):
        env_prim_path = prim_path_template.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_id}")
        env_prim_path = re.sub(r"env_\.\*", f"env_{env_id}", env_prim_path)
        shader_path = f"{env_prim_path}/geometry/material/Shader"
        shader_prim = scene.stage.GetPrimAtPath(shader_path)
        if not shader_prim.IsValid():
            raise RuntimeError(f"Target cube shader prim not found: {shader_path}")

        shader = UsdShade.Shader(shader_prim)
        color_input = shader.GetInput("diffuseColor")
        if not color_input:
            raise RuntimeError(f"Target cube diffuseColor input not found: {shader_path}")

        color_input.Set(Gf.Vec3f(*color_rgb))
