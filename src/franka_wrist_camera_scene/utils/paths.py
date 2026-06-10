"""Relative paths resolution, config path helpers, and dataset output path builders."""

from __future__ import annotations

from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]


def get_config_path(config_name: str) -> Path:
    """Return the absolute path to a configuration file in the configs/ directory."""
    return REPO_ROOT / "configs" / config_name


def load_yaml_config(config_name: str) -> dict:
    """Load and return a YAML configuration file as a dict."""
    config_path = get_config_path(config_name)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
