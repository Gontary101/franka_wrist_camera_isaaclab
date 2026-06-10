"""Lightweight viewport markers for the circle-drawing motion."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg


@dataclass(slots=True)
class CircleMotionMarkers:
    """Visualize the commanded circle and the moving IK target."""

    root_prim_path: str = "/Visuals/franka_circle_motion"
    path_radius_m: float = 0.006
    target_radius_m: float = 0.025
    _path: VisualizationMarkers = field(init=False, repr=False)
    _target: VisualizationMarkers = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._path = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path=f"{self.root_prim_path}/path",
                markers={
                    "point": sim_utils.SphereCfg(
                        radius=self.path_radius_m,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.65, 1.0)),
                    )
                },
            )
        )
        self._target = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path=f"{self.root_prim_path}/target",
                markers={
                    "target": sim_utils.SphereCfg(
                        radius=self.target_radius_m,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.55, 0.0)),
                    )
                },
            )
        )

    def draw_path(self, points_w: torch.Tensor) -> None:
        """Draw the desired circle as small point instances in world coordinates."""
        self._path.visualize(translations=points_w)

    def draw_target(self, position_w: torch.Tensor) -> None:
        """Draw the instantaneous IK target position in world coordinates."""
        self._target.visualize(translations=position_w)
