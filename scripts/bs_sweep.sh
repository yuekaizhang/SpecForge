#!/bin/bash
# LibriSpeech test-clean concurrency sweep (c=2/4/8/16) against one sglang server.
# Usage: bs_sweep.sh <tag> <gpu> <port> [spec args...]
# Serves with the DEFAULT cuda-graph config (no --cuda-graph-max-bs-decode cap)
# to probe whether full-graph capture + spec decoding OOMs at mem-fraction 0.85.
set -uo pipefail
TAG=$1; GPU=$2; PORT=$3; shift 3
ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative
PY=$ROOT/.specforge_env/bin/python
cd $ROOT/SpecForge
SLOG=/tmp/serve_sweep_${TAG}.log

# setsid: own process group so we can kill the whole server tree without
# touching this script or the concurrent sweeps on other GPUs.
CUDA_VISIBLE_DEVICES=$GPU SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/.cache/huggingface \
setsid $PY -m sglang.launch_server --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --served-model-name qwen3-omni \
  --tp 1 --host 127.0.0.1 --port $PORT --dtype bfloat16 --attention-backend fa3 \
  --mem-fraction-static 0.85 --grammar-backend none "$@" > $SLOG 2>&1 &
SPID=$!

for i in $(seq 1 240); do
  grep -qm1 "fired up" $SLOG && break
  if ! kill -0 $SPID 2>/dev/null || grep -qE "Traceback|CUDA out of memory" $SLOG; then
    echo "RESULT $TAG SERVER_FAILED (see $SLOG)"
    grep -m1 -E "CUDA out of memory|RuntimeError|Traceback" $SLOG
    exit 1
  fi
  sleep 10
done
echo "RESULT $TAG server ready"

for C in 2 4 8 16; do
  N=$(grep -c "accept len" $SLOG || true)
  OUT=results/omni_libri_sweep/${TAG}_c${C}
  HF_HOME=/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/.cache/huggingface \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  $PY scripts/decode_sglang.py --server http://127.0.0.1:$PORT --model qwen3-omni \
    --dataset openslr/librispeech_asr --subset clean --split test --text-column text \
    --prompt "Transcribe the English audio into text." --normalize-english \
    --concurrency $C --output-dir $OUT > /tmp/decode_sweep_${TAG}_c${C}.log 2>&1
  RC=$?
  if [ $RC -ne 0 ] || [ ! -f $OUT/summary.txt ]; then
    echo "RESULT $TAG c$C DECODE_FAILED rc=$RC"
    grep -m1 -E "CUDA out of memory|Traceback" $SLOG
    continue
  fi
  WALL=$(grep "Wall time" $OUT/summary.txt | grep -o "[0-9.]*")
  WER=$(grep "^WER" $OUT/summary.txt | grep -o "[0-9.]*%" | head -1)
  THR=$(grep "Throughput" $OUT/summary.txt | grep -o "[0-9.]*")
  ACC=$(grep -o "accept len: [0-9.]*" $SLOG | tail -n +$((N+1)) | awk '{s+=$3; n++} END {if(n>0) printf "%.2f", s/n; else printf "-"}')
  echo "RESULT $TAG c$C wall=${WALL}s thr=${THR}utt/s wer=$WER accept=$ACC"
done
echo "RESULT $TAG sweep done"
kill -9 -- -$SPID 2>/dev/null || kill -9 $SPID 2>/dev/null
