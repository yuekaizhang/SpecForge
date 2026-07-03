#!/bin/bash
# Single-node 8xH100 Slurm launcher for Qwen3-Omni-30B EAGLE3 draft training
# on LibriSpeech clean/train.100 (regenerated greedy labels), FROM SCRATCH,
# 50 epochs @ lr 1e-5. Runs the torchrun launcher
# (examples/run_qwen3_omni_eagle3.sh) INSIDE the .sqsh container via
# pyxis/enroot. NOT a Ray job — torchrun spawns the 8 local ranks itself.
#
# Submit from the LOGIN node (USER=yuekaiz):
#   sbatch examples/sbatch_qwen3_omni_eagle3.sh
#
# Timing: ~3568 steps/epoch @ ~0.2 s/step ≈ 12-13 min/epoch -> 50 epochs
# ≈ 10-11 h + ~10 min startup per chunk. With --time=4:00:00 and
# --dependency=singleton, submit this script ~3 times back-to-back (or
# resubmit after each chunk): RESUME=1 is safe — fresh start when OUTPUT_DIR
# has no checkpoint, resume from the latest checkpoint otherwise.
#
# NOTE: wandb online needs WANDB_API_KEY in the submitting shell
# (srun --export=ALL propagates it). If absent, set WANDB_OFFLINE=1.
# The 28539-row processed-dataset + vocab-mapping caches are warm on /lustre
# (built by the interactive run), so batch jobs skip the mel extraction.
#
#SBATCH --job-name=qwen3omni-eagle3
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
NUM_EPOCHS=${NUM_EPOCHS:-50}
LR=${LR:-1e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.003}
RESUME=${RESUME:-1}          # safe: fresh if no checkpoint, else resume (chaining)
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen3-omni-30b-eagle3-50ep}
WANDB_NAME=${WANDB_NAME:-librispeech-train100-50ep}
TTT_LENGTH=${TTT_LENGTH:-5}
TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}
LABELS=${LABELS:-$ROOT/SpecForge/outputs/librispeech_clean_train100_qwen3omni_labels.jsonl}

mkdir -p "$ROOT/logs/slurm" "$ROOT/SpecForge/$OUTPUT_DIR"

# The launcher self-activates .specforge_env (transformers 5.8.1 / torch 2.11)
# from /lustre and runs fully offline (HF_HUB_OFFLINE inside the launcher).
srun \
  --container-image="$CONTAINER" \
  --container-mounts=/lustre:/lustre \
  --no-container-mount-home \
  --export=ALL \
  bash -lc "cd $ROOT/SpecForge && \
    NUM_EPOCHS=$NUM_EPOCHS LR=$LR WARMUP_RATIO=$WARMUP_RATIO RESUME=$RESUME \
    TTT_LENGTH=$TTT_LENGTH TARGET_MODEL=$TARGET_MODEL LABELS=$LABELS \
    OUTPUT_DIR=$OUTPUT_DIR WANDB_NAME=$WANDB_NAME \
    bash examples/run_qwen3_omni_eagle3.sh"
