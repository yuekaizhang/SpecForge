#!/bin/bash
# Qwen3-Omni-30B-A3B-Instruct (thinker) EAGLE3 draft training on LibriSpeech
# clean/train.100 with regenerated greedy labels
# (outputs/librispeech_clean_train100_qwen3omni_labels.jsonl, from
# scripts/generate_response.py). Mirrors the proven run_qwen2_audio_eagle3
# recipe: 8-GPU pure DP (one full ~59 GiB bf16 thinker per H100), batch 1,
# sdpa, reference loss, dynamo off.
#
#   full run:   bash examples/run_qwen3_omni_eagle3.sh
#   smoke:      SMOKE=1 bash examples/run_qwen3_omni_eagle3.sh
#   resume:     RESUME=1 bash examples/run_qwen3_omni_eagle3.sh
set -euo pipefail
ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative
source "$ROOT/.specforge_env/bin/activate"
cd "$ROOT/SpecForge"
# Model + openslr/librispeech_asr are cached; run fully offline (compute nodes
# flake on HF Hub, and the label idx-join requires the exact cached dataset).
export HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/.cache/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Proven env workarounds (OMP/MKL=1: fork+OpenMP torch.stft deadlock in
# datasets.map workers; expandable_segments: TTT-unroll allocator churn):
export TORCHDYNAMO_DISABLE=1
export SPECFORGE_REFERENCE_LOSS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_EPOCHS=${NUM_EPOCHS:-2}
LR=${LR:-1e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.003}
RESUME_ARG=""; [ "${RESUME:-0}" = "1" ] && RESUME_ARG="--resume"
OUTPUT_DIR=${OUTPUT_DIR:-outputs/qwen3-omni-30b-eagle3}
WANDB_NAME=${WANDB_NAME:-librispeech-train100-ep${NUM_EPOCHS}}
DIST_TIMEOUT=${DIST_TIMEOUT:-120}
TARGET_MODEL=${TARGET_MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}
# MUST match the prompt used by scripts/generate_response.py for the labels.
INSTRUCTION="Transcribe the English audio into text."
LABELS=${LABELS:-$ROOT/SpecForge/outputs/librispeech_clean_train100_qwen3omni_labels.jsonl}
TRAIN_SPLIT=${TRAIN_SPLIT:-train.100}
TTT_LENGTH=${TTT_LENGTH:-5}
MAX_STEPS_ARG=""
if [ "${SMOKE:-0}" = "1" ]; then
  # 2 optimizer steps on a 64-row slice; separate output dir + cache key
  # (train-split differs) so the real run stays cold-cache clean.
  TRAIN_SPLIT="train.100[:64]"
  MAX_STEPS_ARG="--max-num-steps 2"
  OUTPUT_DIR=outputs/qwen3-omni-30b-eagle3-smoke
  WANDB_NAME=smoke
fi

torchrun --nproc_per_node 8 scripts/train_eagle3.py \
  --target-model-path "$TARGET_MODEL" \
  --draft-model-config configs/qwen3-omni-30b-eagle3.json \
  --train-data-path openslr/librispeech_asr \
  --train-config clean \
  --train-split "$TRAIN_SPLIT" \
  --text-column text \
  --keep-transcription-spaces \
  --label-override "$LABELS" \
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
  --learning-rate $LR \
  --warmup-ratio $WARMUP_RATIO \
  --ttt-length $TTT_LENGTH \
  --num-epochs $NUM_EPOCHS \
  $RESUME_ARG \
  $MAX_STEPS_ARG \
  --save-interval 1000 \
  --output-dir $OUTPUT_DIR \
  --report-to wandb \
  --wandb-project qwen3omni-eagle3-librispeech \
  --wandb-name "$WANDB_NAME" \
  --wandb-dir $ROOT/wandb
