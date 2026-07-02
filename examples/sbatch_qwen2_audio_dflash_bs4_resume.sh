#!/bin/bash
# Single-node 8xH100 Slurm launcher for SpecForge Qwen2-Audio DFlash training,
# BLOCK_SIZE=4 CONTINUED (resume from the existing block-4 checkpoint at
# outputs/qwen2-audio-sft-dflash-fixedhead). Runs the torchrun launcher inside
# the .sqsh container via pyxis/enroot. Single node, torchrun spawns 8 ranks.
#
# Submit from the LOGIN node (USER=yuekaiz):
#   sbatch examples/sbatch_qwen2_audio_dflash_bs4_resume.sh
# RESUME=1: picks up the latest checkpoint in OUTPUT_DIR (epoch_3_step_42000 now)
# and continues. Note: on resume the draft config (block_size=4) is loaded FROM
# the checkpoint, so this always continues block-4 regardless of DRAFT_CONFIG.
# With --dependency=singleton, resubmit to chain 4h chunks until done.
#
#SBATCH --job-name=qwen2audio-dflash-bs4
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
DRAFT_CONFIG=${DRAFT_CONFIG:-configs/qwen2-audio-7b-dflash-bs4.json}
BLOCK_SIZE=${BLOCK_SIZE:-4}
NUM_EPOCHS=${NUM_EPOCHS:-50}
LR=${LR:-1e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.003}
RESUME=${RESUME:-1}          # continue from the latest checkpoint in OUTPUT_DIR
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen2-audio-sft-dflash-fixedhead}
WANDB_NAME=${WANDB_NAME:-aishell-sft-dflash-bs4-resume}
TARGET_MODEL=${TARGET_MODEL:-$ROOT/.hf_cache/hub/models--yuekai--qwen2_audio_aishell_sft/snapshots/1cbdccf78fb863da86f0049a2930dcf2950bff37}
INSTRUCTION=${INSTRUCTION:-"Detect the language and recognize the speech: <|zh|>"}

mkdir -p "$ROOT/logs/slurm"

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
