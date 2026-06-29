#!/bin/bash
# Full AISHELL train-split, 1 epoch, 8-GPU data-parallel.
# Effective batch = 8 (batch-size 1 per GPU x 8 ranks). lr 1e-4 = repo EAGLE3 standard.
# Checkpoints every ~3000 steps (~hourly). Same env workarounds as the proof run
# (sdpa attention, reference loss, OMP=1, expandable allocator) — see comments in
# run_qwen2_audio_eagle3_online.sh for why.
set -euo pipefail
ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative
# Self-contained: use the isolated SpecForge env (Python 3.12) so `specforge` imports.
source "$ROOT/.specforge_env/bin/activate"
export HF_HOME=$ROOT/.hf_cache
export TORCHDYNAMO_DISABLE=1
export SPECFORGE_REFERENCE_LOSS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Parametrized so the SAME script runs epoch 0 and the resume:
#   epoch 0:   bash run_qwen2_audio_eagle3_full.sh
#   resume:    NUM_EPOCHS=2 RESUME=1 bash run_qwen2_audio_eagle3_full.sh
# --resume picks up the latest checkpoint under --output-dir and restores
# optimizer/scheduler/epoch/step (true continue, not finetune-from-scratch).
NUM_EPOCHS=${NUM_EPOCHS:-1}
RESUME_ARG=""; [ "${RESUME:-0}" = "1" ] && RESUME_ARG="--resume"
WANDB_NAME=${WANDB_NAME:-aishell-full-ep${NUM_EPOCHS}}
# Online by default (uploads to wandb.ai; needs WANDB_API_KEY in env, which is set).
# Set WANDB_OFFLINE=1 to log locally only (then `wandb sync $ROOT/wandb` later).
WANDB_OFFLINE_ARG=""; [ "${WANDB_OFFLINE:-0}" = "1" ] && WANDB_OFFLINE_ARG="--wandb-offline"
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen2-audio-7b-eagle3-full}
# rank-0 builds the (large, single-proc) dataset cache while other ranks wait at
# the rank_0_priority barrier; that build can take ~30min, exceeding the default
# 20min collective timeout. Give a generous timeout so the first run (cold cache)
# does not time out. Subsequent runs hit the warm cache and pass instantly.
DIST_TIMEOUT=${DIST_TIMEOUT:-120}
TARGET_MODEL=${TARGET_MODEL:-yuekai/qwen2_audio_aishell_sft}
INSTRUCTION=${INSTRUCTION:-"Detect the language and recognize the speech: <|zh|>"}
LR=${LR:-1e-5}  # best from sweep
WARMUP_RATIO=${WARMUP_RATIO:-0.003}  # best from sweep (0.003)
# W&B in offline mode (container may not reach api.wandb.ai); `wandb sync $ROOT/wandb`
# from the login node later. Override project/name via WANDB_NAME env.
INSTRUCTION_ARG=""; [ -n "$INSTRUCTION" ] && INSTRUCTION_ARG="--instruction $INSTRUCTION"
torchrun --nproc_per_node 8 scripts/train_eagle3.py \
  --target-model-path $TARGET_MODEL \
  --draft-model-config configs/qwen2-audio-7b-eagle3.json \
  --train-data-path carlot/AIShell \
  --train-split train \
  --is-audio \
  $INSTRUCTION_ARG \
  --chat-template qwen \
  --build-dataset-num-proc 1 \
  --target-model-backend custom \
  --embedding-key language_model.model.embed_tokens.weight \
  --attention-backend sdpa \
  --dist-timeout $DIST_TIMEOUT \
  --batch-size 1 \
  --max-length 768 \
  --learning-rate $LR \
  --warmup-ratio $WARMUP_RATIO \
  --ttt-length 5 \
  --num-epochs $NUM_EPOCHS \
  $RESUME_ARG \
  --report-to wandb \
  $WANDB_OFFLINE_ARG \
  --wandb-project qwen2audio-eagle3-aishell \
  --wandb-name "$WANDB_NAME" \
  --wandb-dir $ROOT/wandb \
  --eval-data-path carlot/AIShell \
  --eval-split "validation[:500]" \
  --eval-interval 2000 \
  --save-interval 2000 \
  --output-dir $OUTPUT_DIR
