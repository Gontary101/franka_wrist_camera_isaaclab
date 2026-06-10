"""Base definitions for Franka tabletop tasks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Base task specification containing the language instruction."""

    instruction: str
