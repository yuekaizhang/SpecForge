#!/bin/bash
# Qwen3-Omni-30B DFlash draft training on the MULTI-DOMAIN mix (~500k utts,
# regenerated labels baked into outputs/multidomain/combined_train), block_size=6.
# Reuses the SAME preprocessed cache as the EAGLE3 multidomain run (cache key
# aligned), so startup skips the ~3 h mel preprocessing.
#
#   full run:  bash examples/run_qwen3_omni_dflash_multidomain.sh
#   smoke:     SMOKE=1 bash examples/run_qwen3_omni_dflash_multidomain.sh
#   resume:    RESUME=1 bash examples/run_qwen3_omni_dflash_multidomain.sh
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
# Resume fast-forward can be NCCL-silent for >480 s on 60k-step epochs.
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600

NUM_EPOCHS=${NUM_EPOCHS:-50}
LR=${LR:-1e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.003}
BLOCK_SIZE=${BLOCK_SIZE:-6}
RESUME_ARG=""; [ "${RESUME:-0}" = "1" ] && RESUME_ARG="--resume"
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen3-omni-30b-dflash-multidomain-bs6}
WANDB_NAME=${WANDB_NAME:-multidomain-500k-dflash-bs6-ep${NUM_EPOCHS}}
TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}
# MUST match the prompt used for label generation (generate_response.py).
INSTRUCTION="Transcribe the English audio into text."
TRAIN_DATA=${TRAIN_DATA:-outputs/multidomain/combined_train}
MAX_AUDIO_FRAMES=${MAX_AUDIO_FRAMES:-3000}
DIST_TIMEOUT=${DIST_TIMEOUT:-14400}
MAX_STEPS_ARG=""
if [ "${SMOKE:-0}" = "1" ]; then
  TRAIN_DATA=outputs/multidomain/combined_train_smoke64
  MAX_STEPS_ARG="--max-num-steps 2"
  OUTPUT_DIR=outputs/qwen3-omni-30b-dflash-multidomain-smoke
  WANDB_NAME=dflash-md-smoke
  DIST_TIMEOUT=600
fi

torchrun --nproc_per_node 8 scripts/train_dflash.py \
  --target-model-path "$TARGET_MODEL" \
  --draft-config-path configs/qwen3-omni-30b-dflash-bs6.json \
  --train-data-path "$TRAIN_DATA" \
  --keep-transcription-spaces \
  --is-audio \
  --instruction "$INSTRUCTION" \
  --chat-template qwen \
  --target-model-backend hf \
  --embedding-key thinker.model.embed_tokens.weight \
  --lm-head-key thinker.lm_head.weight \
  --mask-token-id 151662 \
  --attention-backend sdpa \
  --build-dataset-num-proc 1 \
  --dist-timeout $DIST_TIMEOUT \
  --batch-size 1 \
  --max-length 1024 \
  --max-audio-frames $MAX_AUDIO_FRAMES \
  --block-size $BLOCK_SIZE \
  --learning-rate $LR \
  --warmup-ratio $WARMUP_RATIO \
  --num-epochs $NUM_EPOCHS \
  $RESUME_ARG \
  $MAX_STEPS_ARG \
  --save-interval ${SAVE_INTERVAL:-8000} \
  --output-dir $OUTPUT_DIR \
  --report-to wandb \
  --wandb-project qwen3omni-dflash-multidomain \
  --wandb-name "$WANDB_NAME" \
  --wandb-dir $ROOT/wandb
