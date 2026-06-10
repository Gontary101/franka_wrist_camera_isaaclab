#!/usr/bin/env bash
# Helper script to run the Franka wrist camera simulation with pre-configured env variables.

# Exit immediately if a command exits with a non-zero status
set -euo pipefail

# Get directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Clean up conda env variables to prevent conflicting python environment paths
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV

# Set PYTHONPATH to include all relevant Isaac Lab modules and the project src folder
export PYTHONPATH="/home/utilisateur/IsaacLab/source/isaaclab:/home/utilisateur/IsaacLab/source/isaaclab_assets:/home/utilisateur/IsaacLab/source/isaaclab_contrib:/home/utilisateur/IsaacLab/source/isaaclab_mimic:/home/utilisateur/IsaacLab/source/isaaclab_rl:/home/utilisateur/IsaacLab/source/isaaclab_tasks:${SCRIPT_DIR}/src:${PYTHONPATH:-}"

# Force Vulkan to use the NVIDIA ICD (prevents interference from integrated graphics GPUs)
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export TERM=xterm

# Execute the simulation run script passing along all arguments
exec "${SCRIPT_DIR}/scripts/run.sh" "$@"
