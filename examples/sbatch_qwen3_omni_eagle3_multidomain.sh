#!/bin/bash
# Single-node 8xH100 Slurm launcher for Qwen3-Omni-30B EAGLE3 draft training on
# the MULTI-DOMAIN mix (~500k utts: GigaSpeech-s / AMI-ihm / SPGISpeech-S /
# Earnings22 / LibriSpeech-100 / VoxPopuli-4k; regenerated labels baked into
# outputs/multidomain/combined_train, globally shuffled), ttt7 recipe.
#
# Submit from the LOGIN node (USER=yuekaiz):
#   sbatch examples/sbatch_qwen3_omni_eagle3_multidomain.sh
#
# Timing: ~500k utts / 8 GPUs ≈ 62.6k steps/epoch @ ~0.25 s/step ≈ 4.5 h/epoch.
# NUM_EPOCHS=50 default ≈ 220 h of training — stop the chain whenever accept
# plateaus (checkpoints every 8000 steps); ~1 chunk ≈ 1 epoch after warm cache.
# FIRST cold-cache chunk spends ~3 h on rank-0 single-proc mel preprocessing
# (dist-timeout is set to cover it; the cache persists on /lustre for all later
# chunks). With --time=4:00:00 + --dependency=singleton, resubmit this script
# repeatedly to chain chunks: RESUME=1 is fresh-start-safe and resumes from the
# latest checkpoint otherwise.
#
# NOTE: wandb online needs WANDB_API_KEY in the submitting shell
# (srun --export=ALL propagates it).
#
#SBATCH --job-name=qwen3omni-eagle3-md
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
RESUME=${RESUME:-1}          # fresh if no checkpoint, else resume (chaining)
TTT_LENGTH=${TTT_LENGTH:-7}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen3-omni-30b-eagle3-multidomain-ttt7}
WANDB_NAME=${WANDB_NAME:-multidomain-500k-ttt7-ep${NUM_EPOCHS}}
TRAIN_DATA=${TRAIN_DATA:-outputs/multidomain/combined_train}
DIST_TIMEOUT=${DIST_TIMEOUT:-14400}
SAVE_INTERVAL=${SAVE_INTERVAL:-8000}

mkdir -p "$ROOT/logs/slurm" "$ROOT/SpecForge/$OUTPUT_DIR"

srun \
  --container-image="$CONTAINER" \
  --container-mounts=/lustre:/lustre \
  --no-container-mount-home \
  --export=ALL \
  bash -lc "cd $ROOT/SpecForge && \
    NUM_EPOCHS=$NUM_EPOCHS LR=$LR WARMUP_RATIO=$WARMUP_RATIO RESUME=$RESUME \
    TTT_LENGTH=$TTT_LENGTH TRAIN_DATA=$TRAIN_DATA DIST_TIMEOUT=$DIST_TIMEOUT SAVE_INTERVAL=$SAVE_INTERVAL \
    OUTPUT_DIR=$OUTPUT_DIR WANDB_NAME=$WANDB_NAME \
    bash examples/run_qwen3_omni_eagle3_multidomain.sh"
