"""Collection manifest writer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EpisodeManifestEntry:
    episode_id: int
    episode_dir: str
    success: bool
    num_steps: int
    num_camera_frames: int
    object_pos_local: tuple[float, float, float] | None
    place_pos_local: tuple[float, float, float] | None
    seed: int | None
    object_xy_offset: tuple[float, float] | None
    place_xy_offset: tuple[float, float] | None
    object_color_name: str | None
    object_color_rgb: tuple[float, float, float] | None
    trajectory_file: str
    metadata_file: str


@dataclass(frozen=True, slots=True)
class CollectionManifest:
    format_version: int
    task_name: str
    num_episodes: int
    successes: int
    failures: int
    episodes: list[EpisodeManifestEntry]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def write_collection_manifest(
    output_dir: Path,
    task_name: str,
    episode_dirs: list[Path],
) -> Path:
    entries: list[EpisodeManifestEntry] = []

    for episode_dir in sorted(episode_dirs):
        meta_path = episode_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        rel_dir = episode_dir.relative_to(output_dir)

        entries.append(
            EpisodeManifestEntry(
                episode_id=int(meta["episode_id"]),
                episode_dir=rel_dir.as_posix(),
                success=bool(meta["success"]),
                num_steps=int(meta["num_steps"]),
                num_camera_frames=int(meta.get("num_camera_frames", 0)),
                object_pos_local=tuple(meta["object_pos_local"]) if meta.get("object_pos_local") is not None else None,
                place_pos_local=tuple(meta["place_pos_local"]) if meta.get("place_pos_local") is not None else None,
                seed=meta.get("seed"),
                object_xy_offset=tuple(meta["object_xy_offset"]) if meta.get("object_xy_offset") is not None else None,
                place_xy_offset=tuple(meta["place_xy_offset"]) if meta.get("place_xy_offset") is not None else None,
                object_color_name=meta.get("object_color_name"),
                object_color_rgb=tuple(meta["object_color_rgb"]) if meta.get("object_color_rgb") is not None else None,
                trajectory_file=(rel_dir / "trajectory.npz").as_posix(),
                metadata_file=(rel_dir / "meta.json").as_posix(),
            )
        )

    successes = sum(entry.success for entry in entries)

    manifest = CollectionManifest(
        format_version=1,
        task_name=task_name,
        num_episodes=len(entries),
        successes=successes,
        failures=len(entries) - successes,
        episodes=entries,
    )

    manifest_path = output_dir / "manifest.json"
    manifest.save(manifest_path)
    return manifest_path
