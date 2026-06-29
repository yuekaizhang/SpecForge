#!/bin/bash
# Single-node 8xH100 Slurm launcher for SpecForge Qwen2-Audio EAGLE3 training.
# Runs the torchrun launcher (examples/run_qwen2_audio_eagle3_full.sh) INSIDE the
# given .sqsh container via pyxis/enroot. NOT a Ray job — single node, torchrun
# spawns the 8 local ranks itself.
#
# Submit from the LOGIN node (USER=yuekaiz):
#   sbatch examples/sbatch_qwen2_audio_eagle3.sh
# Override per-run knobs at submit time, e.g.:
#   LR=5e-5 NUM_EPOCHS=10 RESUME=0 sbatch examples/sbatch_qwen2_audio_eagle3.sh
#
#SBATCH --job-name=qwen2audio-eagle3
#SBATCH --account=coreai_dlalgo_nemorl
#SBATCH --partition=batch
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --dependency=singleton
#SBATCH --output=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative/logs/slurm/slurm-%j.out
#SBATCH --error=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative/logs/slurm/slurm-%j.err

set -euo pipefail

ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative
CONTAINER=/lustre/fsw/portfolios/coreai/users/yuekaiz/containers/nemo_rl.0615.rebuild.sqsh

# --- per-run knobs (override via env at submit) ---
# Defaults: SFT target model + matched prompt (100% GT alignment, no train/infer mismatch)
TARGET_MODEL=${TARGET_MODEL:-yuekai/qwen2_audio_aishell_sft}
INSTRUCTION=${INSTRUCTION:-"Detect the language and recognize the speech: <|zh|>"}
LR=${LR:-1e-4}
WARMUP_RATIO=${WARMUP_RATIO:-0.015}
NUM_EPOCHS=${NUM_EPOCHS:-10}
RESUME=${RESUME:-0}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen2-audio-sft-eagle3}
WANDB_NAME=${WANDB_NAME:-aishell-sft-eagle3-10ep}
# wandb online by default (needs WANDB_API_KEY in the submitting env so srun
# propagates it). If the key is absent, set WANDB_OFFLINE=1 to log to $ROOT/wandb.
WANDB_OFFLINE=${WANDB_OFFLINE:-0}

mkdir -p "$ROOT/logs/slurm"

# The training launcher self-activates the .specforge_env venv (Python 3.12,
# torch 2.9.1 / transformers 4.57.1 / sglang 0.5.9) which lives on /lustre, so it
# is available inside the container. The clean (space-stripped) dataset + vocab
# cache are warm on /lustre, so this skips the ~30min build and starts training.
srun \
  --container-image="$CONTAINER" \
  --container-mounts=/lustre:/lustre \
  --no-container-mount-home \
  --export=ALL \
  bash -lc "cd $ROOT/SpecForge && \
    TARGET_MODEL=$TARGET_MODEL INSTRUCTION='$INSTRUCTION' \
    LR=$LR WARMUP_RATIO=$WARMUP_RATIO NUM_EPOCHS=$NUM_EPOCHS RESUME=$RESUME \
    OUTPUT_DIR=$OUTPUT_DIR WANDB_NAME=$WANDB_NAME WANDB_OFFLINE=$WANDB_OFFLINE \
    bash examples/run_qwen2_audio_eagle3_full.sh"
