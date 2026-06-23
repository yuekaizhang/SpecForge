#!/bin/bash
set -euo pipefail
export HF_HOME=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative/.hf_cache
export TORCHDYNAMO_DISABLE=1
export SPECFORGE_REFERENCE_LOSS=1
# Audio preprocessing runs the Qwen2-Audio (Whisper) feature extractor, which
# calls torch.stft. The target model is loaded on CUDA before dataset.map, so
# torch's intra-op (OpenMP) thread pool is already initialized; datasets.map
# forks a worker process that inherits the poisoned thread pool and then
# deadlocks inside torch.stft (classic fork + OpenMP-threadpool hang).
# Forcing single-threaded math in every process (OMP/MKL=1) removes the
# OpenMP barrier that deadlocks across the fork. --build-dataset-num-proc 1
# also keeps preprocessing to a single worker.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
# Reduce allocator fragmentation; the draft TTT unroll allocates/frees large
# vocab-sized logit tensors each step, so expandable segments help avoid OOM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --nproc_per_node 1 scripts/train_eagle3.py \
  --target-model-path Qwen/Qwen2-Audio-7B-Instruct \
  --draft-model-config configs/qwen2-audio-7b-eagle3.json \
  --train-data-path /tmp/aishell_train3k \
  --is-audio \
  --chat-template qwen \
  --build-dataset-num-proc 1 \
  --target-model-backend custom \
  --embedding-key language_model.model.embed_tokens.weight \
  --attention-backend sdpa \
  --batch-size 1 \
  --max-length 768 \
  --learning-rate 1e-4 \
  --ttt-length 5 \
  --num-epochs 1 \
  --max-num-steps 300 \
  --save-interval 300 \
  --output-dir outputs/qwen2-audio-7b-eagle3
