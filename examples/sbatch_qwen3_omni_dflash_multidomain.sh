#!/bin/bash
# Single-node 8xH100 Slurm launcher for Qwen3-Omni-30B DFlash draft training on
# the MULTI-DOMAIN mix (block_size=6, min 7 loss tokens, max 3000 mel frames ->
# 422,800 samples). RESUMES from the latest checkpoint in OUTPUT_DIR
# (currently epoch_1_step_96000, trained interactively to ~1.88 epochs;
# LibriSpeech accept 3.70 / 1.52x at that point).
#
# Submit from the LOGIN node (USER=yuekaiz):
#   sbatch examples/sbatch_qwen3_omni_dflash_multidomain.sh
#
# Timing: ~52,850 steps/epoch @ ~6.5 it/s ≈ 2h15m/epoch. All caches are warm on
# /lustre (138.6 GB shared preprocessed cache + filtered indices), so every
# chunk starts training within ~10 min. With --time=4:00:00 +
# --dependency=singleton, resubmit repeatedly to chain 4h chunks (~1.5 epochs
# per chunk); RESUME=1 is fresh-start-safe and resumes otherwise. Stop the
# chain when LibriSpeech accept plateaus (checkpoints every 8000 steps).
# NOTE: resume uses the fresh-optimizer fallback (LR scheduler restored,
# Adam momentum re-warms) — known-good from the Qwen2-Audio DFlash runs.
#
# wandb online needs WANDB_API_KEY in the submitting shell (--export=ALL).
#
#SBATCH --job-name=qwen3omni-dflash-md
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
BLOCK_SIZE=${BLOCK_SIZE:-6}
RESUME=${RESUME:-1}          # resume from latest ckpt in OUTPUT_DIR (fresh-safe)
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen3-omni-30b-dflash-multidomain-bs6}
WANDB_NAME=${WANDB_NAME:-multidomain-500k-dflash-bs6-ep${NUM_EPOCHS}}
TRAIN_DATA=${TRAIN_DATA:-outputs/multidomain/combined_train}
MAX_AUDIO_FRAMES=${MAX_AUDIO_FRAMES:-3000}
SAVE_INTERVAL=${SAVE_INTERVAL:-8000}
DIST_TIMEOUT=${DIST_TIMEOUT:-14400}

mkdir -p "$ROOT/logs/slurm" "$ROOT/SpecForge/$OUTPUT_DIR"

srun \
  --container-image="$CONTAINER" \
  --container-mounts=/lustre:/lustre \
  --no-container-mount-home \
  --export=ALL \
  bash -lc "cd $ROOT/SpecForge && \
    NUM_EPOCHS=$NUM_EPOCHS LR=$LR WARMUP_RATIO=$WARMUP_RATIO RESUME=$RESUME \
    BLOCK_SIZE=$BLOCK_SIZE TRAIN_DATA=$TRAIN_DATA MAX_AUDIO_FRAMES=$MAX_AUDIO_FRAMES \
    SAVE_INTERVAL=$SAVE_INTERVAL DIST_TIMEOUT=$DIST_TIMEOUT \
    OUTPUT_DIR=$OUTPUT_DIR WANDB_NAME=$WANDB_NAME \
    bash examples/run_qwen3_omni_dflash_multidomain.sh"
