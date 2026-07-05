#!/bin/bash
# Qwen3-Omni-30B EAGLE3 draft training on the MULTI-DOMAIN mix (~500k utts:
# GigaSpeech-s / AMI-ihm / SPGISpeech-S / Earnings22 / LibriSpeech-100 / VoxPopuli-4k),
# ttt7 recipe. Labels are regenerated greedy outputs BAKED INTO the dataset's
# 'transcription' column (outputs/multidomain/combined_train, globally shuffled)
# — no --label-override / idx alignment needed.
#
#   full run:  bash examples/run_qwen3_omni_eagle3_multidomain.sh
#   smoke:     SMOKE=1 bash examples/run_qwen3_omni_eagle3_multidomain.sh
#   resume:    RESUME=1 bash examples/run_qwen3_omni_eagle3_multidomain.sh
set -euo pipefail
ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative
source "$ROOT/.specforge_env/bin/activate"
cd "$ROOT/SpecForge"
export HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/.cache/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TORCHDYNAMO_DISABLE=1
export SPECFORGE_REFERENCE_LOSS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# On resume, the dataloader fast-forwards (skips) tens of thousands of batches to
# reach the resume step, which issues no NCCL collective for ~10+ min and trips
# the default 480 s NCCL heartbeat watchdog, aborting the job during fast-forward.
# Raise the timeout well past the longest fast-forward (one epoch skip) so resume
# chunks survive; still catches genuine multi-hour deadlocks.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

NUM_EPOCHS=${NUM_EPOCHS:-3}
LR=${LR:-1e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.003}
RESUME_ARG=""; [ "${RESUME:-0}" = "1" ] && RESUME_ARG="--resume"
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen3-omni-30b-eagle3-multidomain-ttt7}
WANDB_NAME=${WANDB_NAME:-multidomain-500k-ttt7-ep${NUM_EPOCHS}}
TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}
# MUST match the prompt used for label generation (generate_response.py).
INSTRUCTION="Transcribe the English audio into text."
TRAIN_DATA=${TRAIN_DATA:-outputs/multidomain/combined_train}
TTT_LENGTH=${TTT_LENGTH:-7}
# Drop very long utterances (AMI/Earnings22) whose variable-length Qwen3-Omni
# mels (~100 frames/s) blow up per-sample memory and OOM at batch-size 1.
# 3000 frames ≈ 30 s, matching the LibriSpeech length that trained fine.
MAX_AUDIO_FRAMES=${MAX_AUDIO_FRAMES:-3000}
# First cold-cache run: rank-0 preprocesses ~500k utts single-proc (~3h) while
# other ranks wait at the barrier — dist timeout must cover it.
DIST_TIMEOUT=${DIST_TIMEOUT:-14400}
MAX_STEPS_ARG=""
if [ "${SMOKE:-0}" = "1" ]; then
  TRAIN_DATA=outputs/multidomain/combined_train_smoke64
  MAX_STEPS_ARG="--max-num-steps 2"
  OUTPUT_DIR=outputs/qwen3-omni-30b-eagle3-multidomain-smoke
  WANDB_NAME=multidomain-smoke
  DIST_TIMEOUT=600
fi

torchrun --nproc_per_node 8 scripts/train_eagle3.py \
  --target-model-path "$TARGET_MODEL" \
  --draft-model-config configs/qwen3-omni-30b-eagle3.json \
  --train-data-path "$TRAIN_DATA" \
  --keep-transcription-spaces \
  --is-audio \
  --instruction "$INSTRUCTION" \
  --chat-template qwen \
  --build-dataset-num-proc 1 \
  --target-model-backend custom \
  --embedding-key thinker.model.embed_tokens.weight \
  --attention-backend sdpa \
  --dist-timeout $DIST_TIMEOUT \
  --batch-size 1 \
  --max-length 1024 \
  --max-audio-frames $MAX_AUDIO_FRAMES \
  --dataloader-num-workers ${DATALOADER_WORKERS:-0} \
  --learning-rate $LR \
  --warmup-ratio $WARMUP_RATIO \
  --ttt-length $TTT_LENGTH \
  --num-epochs $NUM_EPOCHS \
  $RESUME_ARG \
  $MAX_STEPS_ARG \
  --save-interval ${SAVE_INTERVAL:-8000} \
  --output-dir $OUTPUT_DIR \
  --report-to wandb \
  --wandb-project qwen3omni-eagle3-multidomain \
  --wandb-name "$WANDB_NAME" \
  --wandb-dir $ROOT/wandb
