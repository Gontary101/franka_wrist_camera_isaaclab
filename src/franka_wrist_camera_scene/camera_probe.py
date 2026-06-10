"""Optional wrist-camera image coordinate probe."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from isaaclab.scene import InteractiveScene


@dataclass(slots=True)
class WristCameraProbe:
    """Save a wrist-camera RGB image with one annotated pixel and its depth value."""

    u: int = 320
    v: int = 240
    save_every: int = 0
    output_dir: Path = field(default_factory=lambda: Path("camera_probes"))

    def maybe_save(self, scene: InteractiveScene, step: int) -> None:
        """Save an annotated probe image when the configured period is reached."""
        if self.save_every <= 0 or step % self.save_every != 0:
            return

        from PIL import Image, ImageDraw

        self.output_dir.mkdir(parents=True, exist_ok=True)
        camera = scene["wrist_camera"]
        rgb = camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
        depth = camera.data.output["distance_to_image_plane"][0, ..., 0].detach().cpu().numpy()

        height, width = depth.shape
        u = min(max(self.u, 0), width - 1)
        v = min(max(self.v, 0), height - 1)
        z_m = float(depth[v, u])

        image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        draw.line((u - 12, v, u + 12, v), fill=(255, 0, 0), width=2)
        draw.line((u, v - 12, u, v + 12), fill=(255, 0, 0), width=2)
        draw.text((u + 14, v + 14), f"u={u} v={v} z={z_m:.3f} m", fill=(255, 0, 0))
        image.save(self.output_dir / f"wrist_probe_{step:06d}.png")
