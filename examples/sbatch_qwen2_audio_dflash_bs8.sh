#!/bin/bash
# Single-node 8xH100 Slurm launcher for SpecForge Qwen2-Audio DFlash training,
# BLOCK_SIZE=8 (fresh from scratch). Runs the torchrun launcher
# (examples/run_qwen2_audio_dflash.sh) INSIDE the .sqsh container via pyxis/enroot.
# NOT a Ray job — single node, torchrun spawns the 8 local ranks itself.
#
# Submit from the LOGIN node (USER=yuekaiz):
#   sbatch examples/sbatch_qwen2_audio_dflash_bs8.sh
# RESUME=1 is safe on a fresh run (no checkpoint -> starts from scratch) AND on
# re-submits (picks up the latest checkpoint). With --dependency=singleton you can
# submit this repeatedly to chain 4h chunks until 50 epochs finish.
#
#SBATCH --job-name=qwen2audio-dflash-bs8
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
DRAFT_CONFIG=${DRAFT_CONFIG:-configs/qwen2-audio-7b-dflash-bs8.json}
BLOCK_SIZE=${BLOCK_SIZE:-8}
NUM_EPOCHS=${NUM_EPOCHS:-50}
LR=${LR:-1e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.003}
RESUME=${RESUME:-1}          # safe: fresh if no checkpoint, else resume
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen2-audio-sft-dflash-bs8}
WANDB_NAME=${WANDB_NAME:-aishell-sft-dflash-bs8-50ep}
TARGET_MODEL=${TARGET_MODEL:-$ROOT/.hf_cache/hub/models--yuekai--qwen2_audio_aishell_sft/snapshots/1cbdccf78fb863da86f0049a2930dcf2950bff37}
INSTRUCTION=${INSTRUCTION:-"Detect the language and recognize the speech: <|zh|>"}

mkdir -p "$ROOT/logs/slurm" "$ROOT/SpecForge/$OUTPUT_DIR"

# .specforge_env venv (Python 3.12, transformers 5.8.1 / torch 2.11 / sglang HEAD)
# lives on /lustre and is self-activated by the launcher. Model + dataset caches
# are warm on /lustre; HF_HUB_OFFLINE=1 avoids compute-node HF-Hub download races.
srun \
  --container-image="$CONTAINER" \
  --container-mounts=/lustre:/lustre \
  --no-container-mount-home \
  --export=ALL \
  bash -lc "cd $ROOT/SpecForge && \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    DRAFT_CONFIG=$DRAFT_CONFIG BLOCK_SIZE=$BLOCK_SIZE \
    TARGET_MODEL=$TARGET_MODEL INSTRUCTION='$INSTRUCTION' \
    LR=$LR WARMUP_RATIO=$WARMUP_RATIO NUM_EPOCHS=$NUM_EPOCHS RESUME=$RESUME \
    OUTPUT_DIR=$OUTPUT_DIR WANDB_NAME=$WANDB_NAME \
    bash examples/run_qwen2_audio_dflash.sh"
