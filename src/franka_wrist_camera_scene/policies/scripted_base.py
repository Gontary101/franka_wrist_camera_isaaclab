"""Base types for scripted demonstrator policies."""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass(frozen=True, slots=True)
class PolicyCommand:
    """Single Cartesian command produced by a scripted policy."""

    target_pos_w: torch.Tensor
    target_quat_w: torch.Tensor
    finger_opening_m: float
    done: bool = False
