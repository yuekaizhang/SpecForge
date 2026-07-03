#!/bin/bash
# Launch 4 sglang servers for Qwen3-Omni-30B-A3B-Instruct, tp=2 each (8x H100),
# ports 30000-30003, for EAGLE3 training-label generation (text-only ASR).
#
# Facts (verified against the local sglang dev-HEAD):
# - sglang loads the THINKER only; the talker/code2wav (audio output) is
#   hardcoded off (qwen3_omni_moe.py: enable_talker=False) -> text-only output.
# - Multimodal + the repo's chat_template.json are auto-enabled; no
#   --chat-template / --enable-multimodal / --trust-remote-code needed.
# - Audio requests must use {"type":"audio_url","audio_url":{"url":"data:...;base64,..."}}.
#
# Usage (inside the GPU container, USER=root):
#   bash examples/launch_qwen3_omni_servers.sh          # launch + wait healthy
#   STOP=1 bash examples/launch_qwen3_omni_servers.sh   # stop all servers
set -uo pipefail

ROOT=/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/speculative
PY=$ROOT/.specforge_env/bin/python
export HF_HOME=${HF_HOME:-/lustre/fsw/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/yuekaiz/.cache/huggingface}

MODEL=${MODEL:-Qwen/Qwen3-Omni-30B-A3B-Instruct}
SERVED_NAME=${SERVED_NAME:-qwen3-omni}
NUM_SERVERS=${NUM_SERVERS:-4}
TP=${TP:-2}
BASE_PORT=${BASE_PORT:-30000}
MEM_FRACTION=${MEM_FRACTION:-0.85}
LOG_DIR=${LOG_DIR:-$ROOT/SpecForge/outputs/qwen3_omni_servers}
mkdir -p "$LOG_DIR"

# Preflight: Triton JIT (KV-cache warmup kernel) compiles C against Python.h.
# The interactive container is periodically reprovisioned, wiping python3.12-dev;
# without it every server dies at startup with a gcc CalledProcessError.
if [ ! -f /usr/include/python3.12/Python.h ]; then
  echo "Python.h missing — installing python3.12-dev (container was reprovisioned)..."
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y python3.12-dev >/dev/null 2>&1
  [ -f /usr/include/python3.12/Python.h ] || { echo "FATAL: could not install python3.12-dev"; exit 1; }
fi

if [ "${STOP:-0}" = "1" ]; then
  echo "Stopping all sglang servers..."
  ps -eo pid,cmd | grep "sglang.launch_server" | grep python | grep -v grep | awk '{print $1}' | xargs -r kill -9
  sleep 5
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  exit 0
fi

echo "Launching $NUM_SERVERS servers (tp=$TP) on ports $BASE_PORT..$((BASE_PORT + NUM_SERVERS - 1))"
for i in $(seq 0 $((NUM_SERVERS - 1))); do
  PORT=$((BASE_PORT + i))
  GPU0=$((i * TP))
  GPUS=$(seq -s, "$GPU0" $((GPU0 + TP - 1)))
  LOG="$LOG_DIR/server_$PORT.log"
  : > "$LOG"  # truncate BEFORE launch: stale crash lines from a previous run
              # must not trip the health loop's crash detection (race)
  echo "  server $i: GPUs $GPUS -> port $PORT (log: $LOG)"
  setsid bash -c "CUDA_VISIBLE_DEVICES=$GPUS TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=$HF_HOME \
    $PY -m sglang.launch_server \
      --model-path $MODEL --served-model-name $SERVED_NAME \
      --tp $TP --host 127.0.0.1 --port $PORT \
      --dtype bfloat16 --attention-backend fa3 \
      --mem-fraction-static $MEM_FRACTION \
      --grammar-backend none \
      > $LOG 2>&1" &
done

echo "Waiting for health (model load ~3-6 min)..."
DEADLINE=$((SECONDS + 900))
declare -A UP
while [ $SECONDS -lt $DEADLINE ]; do
  all_up=1
  for i in $(seq 0 $((NUM_SERVERS - 1))); do
    PORT=$((BASE_PORT + i))
    if [ "${UP[$PORT]:-0}" != "1" ]; then
      if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        UP[$PORT]=1
        echo "  port $PORT HEALTHY (t=${SECONDS}s)"
      else
        all_up=0
        # surface a crash early
        if grep -qaE "Scheduler hit an exception|child failed|CUDA out of memory" "$LOG_DIR/server_$PORT.log" 2>/dev/null; then
          echo "  port $PORT CRASHED — tail of log:"
          grep -avE "Ignore import error|libtorchcodec|FFmpeg" "$LOG_DIR/server_$PORT.log" | tail -8
          exit 1
        fi
      fi
    fi
  done
  [ $all_up -eq 1 ] && break
  sleep 10
done

n_up=0
for i in $(seq 0 $((NUM_SERVERS - 1))); do
  PORT=$((BASE_PORT + i))
  [ "${UP[$PORT]:-0}" = "1" ] && n_up=$((n_up + 1))
done
echo "$n_up/$NUM_SERVERS servers healthy."
[ "$n_up" -eq "$NUM_SERVERS" ] || exit 1
echo "Server addresses: $(for i in $(seq 0 $((NUM_SERVERS - 1))); do printf '127.0.0.1:%d ' $((BASE_PORT + i)); done)"
