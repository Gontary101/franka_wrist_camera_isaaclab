"""Scene visual material utilities."""

from __future__ import annotations

from pxr import Gf, UsdShade
from isaaclab.scene import InteractiveScene


def set_target_cube_color(scene: InteractiveScene, color_rgb: tuple[float, float, float]) -> None:
    """Set the target cube visual material color per episode after reset."""
    if "target_cube" in scene.keys():
        prim_path_template = scene["target_cube"].cfg.prim_path
    else:
        prim_path_template = "{ENV_REGEX_NS}/TargetCube"

    for env_id in range(scene.num_envs):
        env_prim_path = prim_path_template.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_id}")
        shader_path = f"{env_prim_path}/geometry/material/Shader"
        shader_prim = scene.stage.GetPrimAtPath(shader_path)
        if shader_prim.IsValid():
            shader = UsdShade.Shader(shader_prim)
            color_input = shader.GetInput("diffuseColor")
            if color_input.IsValid():
                color_input.Set(Gf.Vec3f(*color_rgb))
