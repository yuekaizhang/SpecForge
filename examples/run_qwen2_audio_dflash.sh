#!/bin/bash
# DFlash speculative decoding training for Qwen2-Audio (SFT target).
# 5-layer draft, block_size=16, 8-GPU data parallel.
set -euo pipefail
ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative
source "$ROOT/.specforge_env/bin/activate"
export HF_HOME=$ROOT/.hf_cache
# Compute nodes rate-limit / flake on HF Hub; set HF_HUB_OFFLINE=1 (model +
# dataset are warm on /lustre) to avoid mid-startup partial-download races.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-0}
export TORCHDYNAMO_DISABLE=1
export SPECFORGE_REFERENCE_LOSS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_EPOCHS=${NUM_EPOCHS:-10}
LR=${LR:-1e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.003}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen2-audio-sft-dflash}
WANDB_NAME=${WANDB_NAME:-aishell-sft-dflash}
RESUME_ARG=""; [ "${RESUME:-0}" = "1" ] && RESUME_ARG="--resume"
DIST_TIMEOUT=${DIST_TIMEOUT:-120}
TARGET_MODEL=${TARGET_MODEL:-yuekai/qwen2_audio_aishell_sft}
INSTRUCTION=${INSTRUCTION:-"Detect the language and recognize the speech: <|zh|>"}
DRAFT_CONFIG=${DRAFT_CONFIG:-configs/qwen2-audio-7b-dflash.json}
BLOCK_SIZE=${BLOCK_SIZE:-6}

torchrun --nproc_per_node 8 scripts/train_dflash.py \
  --target-model-path $TARGET_MODEL \
  --draft-config-path $DRAFT_CONFIG \
  --train-data-path carlot/AIShell --train-split train \
  --is-audio \
  --instruction "$INSTRUCTION" \
  --chat-template qwen \
  --target-model-backend hf \
  --embedding-key language_model.model.embed_tokens.weight \
  --lm-head-key language_model.lm_head.weight \
  --mask-token-id 151650 \
  --attention-backend sdpa \
  --build-dataset-num-proc 1 \
  --dist-timeout $DIST_TIMEOUT \
  --batch-size 1 \
  --max-length 768 \
  --block-size $BLOCK_SIZE \
  --learning-rate $LR \
  --warmup-ratio $WARMUP_RATIO \
  --num-epochs $NUM_EPOCHS \
  $RESUME_ARG \
  --save-interval 2000 \
  --output-dir $OUTPUT_DIR \
  --report-to wandb \
  --wandb-project qwen2audio-dflash-aishell \
  --wandb-name "$WANDB_NAME" \
  --wandb-dir $ROOT/wandb
