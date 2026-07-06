#!/bin/bash
# Single-node 8xH100 Slurm launcher for the 2-LAYER Qwen3-Omni-30B EAGLE3 draft
# on the MULTI-DOMAIN mix, ttt7 recipe. Identical to
# sbatch_qwen3_omni_eagle3_multidomain.sh except DRAFT_CONFIG points at the
# 2-layer draft config (configs/qwen3-omni-30b-eagle3-2layer.json) and the
# output/wandb names carry a -2layer suffix. Layer count is now config-driven
# in SpecForge (specforge/modeling/draft/llama3_eagle.py builds
# config.num_hidden_layers layers; per-layer TTT KV caches) and sglang's
# llama_eagle3.py already supports multi-layer drafts natively (weights are
# saved as layers.0/layers.1, which sglang loads without remap).
#
# Submit from the LOGIN node (USER=yuekaiz):
#   sbatch examples/sbatch_qwen3_omni_eagle3_multidomain_2layer.sh
#
# The preprocessed-data cache key does NOT include the draft config, so this
# run reuses the warm 138.6 GB multidomain cache — every chunk starts training
# within ~10 min. ~62.6k steps/epoch; the extra layer adds only draft-side
# compute (~536M params vs ~511M), so expect ≈5 h/epoch. With --time=4:00:00 +
# --dependency=singleton, resubmit repeatedly to chain 4 h chunks; RESUME=1 is
# fresh-start-safe. Checkpoints every 8000 steps; stop when accept plateaus.
#
# NOTE: wandb online needs WANDB_API_KEY in the submitting shell
# (srun --export=ALL propagates it).
#
#SBATCH --job-name=qwen3omni-eagle3-md-2l
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
DRAFT_CONFIG=${DRAFT_CONFIG:-configs/qwen3-omni-30b-eagle3-2layer.json}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen3-omni-30b-eagle3-multidomain-ttt7-2layer}
WANDB_NAME=${WANDB_NAME:-multidomain-500k-ttt7-2layer-ep${NUM_EPOCHS}}
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
    TTT_LENGTH=$TTT_LENGTH DRAFT_CONFIG=$DRAFT_CONFIG TRAIN_DATA=$TRAIN_DATA \
    DIST_TIMEOUT=$DIST_TIMEOUT SAVE_INTERVAL=$SAVE_INTERVAL \
    OUTPUT_DIR=$OUTPUT_DIR WANDB_NAME=$WANDB_NAME \
    bash examples/run_qwen3_omni_eagle3_multidomain.sh"
