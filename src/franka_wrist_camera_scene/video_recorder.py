"""Optional video recorder for scene cameras."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from isaaclab.scene import InteractiveScene


@dataclass(slots=True)
class VideoRecorder:
    """Record video outputs from scene cameras."""

    enabled: bool = False
    sim_dt: float = 1.0 / 120.0
    fps: int = 30
    video_writers: dict = field(default_factory=dict, init=False)
    recorded_frames: int = field(default=0, init=False)
    record_interval: int = field(default=1, init=False)
    max_record_frames: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.enabled:
            return

        import cv2

        self.record_interval = max(1, int((1.0 / self.sim_dt) / self.fps))
        self.max_record_frames = 20 * self.fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        self.video_writers = {
            "wrist_camera": cv2.VideoWriter("wrist_camera.mp4", fourcc, self.fps, (480, 480)),
            "agent_camera": cv2.VideoWriter("agent_camera.mp4", fourcc, self.fps, (480, 480)),
        }
        print("[INFO] Recording wrist_camera.mp4 and agent_camera.mp4 for 20 seconds.")

    def record_step(self, scene: InteractiveScene, step: int) -> None:
        """Record a frame if the step matches the interval and limit is not exceeded."""
        if not self.video_writers or self.recorded_frames >= self.max_record_frames:
            return

        if step % self.record_interval == 0:
            import cv2

            for camera_name, writer in self.video_writers.items():
                rgb = scene[camera_name].data.output["rgb"][0].detach().cpu().numpy()[..., :3]
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                writer.write(frame)

            self.recorded_frames += 1
            if self.recorded_frames == self.max_record_frames:
                self.close()
                print("[INFO] Saved wrist_camera.mp4 and agent_camera.mp4")

    def close(self) -> None:
        """Release all video writers."""
        for writer in self.video_writers.values():
            writer.release()
        self.video_writers.clear()
