#!/bin/bash
# One Slurm job: runs its arguments as a command on one GPU, offline, with the cluster environment.
# Usage: sbatch [sbatch options] scripts/job.sbatch <command ...>
#   sbatch scripts/job.sbatch unloop eval --config configs/full.toml --condition base
# Run `unloop fetch` on a login node first; compute nodes read only from the cache.
#SBATCH --job-name=unloop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export UNLOOP_OFFLINE=1
# shellcheck disable=SC1091
. scripts/cluster_env.sh
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$@"
