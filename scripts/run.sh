#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$HOME/IsaacLab}"

export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"
exec "$ISAACLAB_ROOT/isaaclab.sh" -p "$REPO_ROOT/scripts/run_scene.py" "$@"
