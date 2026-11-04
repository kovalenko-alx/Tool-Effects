#!/bin/bash
# Start a vLLM OpenAI-compatible server for a base or merged model (optionally with a LoRA adapter) and wait for it.
#   MODEL=<path> [SERVED_NAME=<name>] [PORT=8000] [LORA=<adapter_dir>] [GPU_UTIL=0.85] [MAX_LEN=16384] [TOOL_PARSER=hermes] [EXTRA="..."]
#   source serve/vllm_launch.sh && vllm_start && ... && vllm_stop
# Logs to $VLLM_LOG (default logs/vllm_<port>.log). Uses the module stack's vllm 0.31 (see env/activate.sh).
# Two jobs on one node with the same PORT (jobs 11835787/88, 2026-10-21): the second server never got the port, /health was
# answered by the first one and the S1 eval silently scored the base model. So: wait while the port is held by someone else,
# and after /health check that /v1/models reports *our* model; on a mismatch, stop ours, wait, retry.
_vllm_port_busy() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
_vllm_root() { curl -sf "http://127.0.0.1:${PORT}/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["root"])' 2>/dev/null; }
_vllm_wait_port_free() {
  local t=0
  while _vllm_port_busy "$PORT"; do
    [ $t -eq 0 ] && echo "[vllm] :$PORT is held by another process (another job on $(hostname)?); waiting for it"
    sleep 15; t=$((t+15))
    if [ $t -ge ${VLLM_PORT_WAIT:-3600} ]; then echo "[vllm] :$PORT still busy after ${t}s"; return 1; fi
  done
}
vllm_start() {
  : "${MODEL:?MODEL path required}"
  PORT=${PORT:-8000}; SERVED_NAME=${SERVED_NAME:-$(basename "$MODEL")}; GPU_UTIL=${GPU_UTIL:-0.85}; MAX_LEN=${MAX_LEN:-16384}; TOOL_PARSER=${TOOL_PARSER:-hermes}
  VLLM_LOG=${VLLM_LOG:-logs/vllm_${PORT}.log}; mkdir -p "$(dirname "$VLLM_LOG")"
  local lora_args="" attempt t root
  if [ -n "${LORA:-}" ]; then lora_args="--enable-lora --max-lora-rank 64 --lora-modules ${SERVED_NAME}-lora=${LORA}"; fi
  for attempt in 1 2 3; do
    _vllm_wait_port_free || return 1
    echo "[vllm] starting $MODEL as $SERVED_NAME on :$PORT (log $VLLM_LOG)"
    python -m vllm.entrypoints.cli.main serve "$MODEL" --served-model-name "$SERVED_NAME" --port "$PORT" --host 127.0.0.1 --dtype bfloat16 \
        --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL" --enable-prefix-caching \
        --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" $lora_args ${EXTRA:-} >> "$VLLM_LOG" 2>&1 &
    VLLM_PID=$!
    export VLLM_PID PORT SERVED_NAME
    t=0
    until curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
      sleep 5; t=$((t+5))
      if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        if _vllm_port_busy "$PORT"; then echo "[vllm] died while :$PORT is held by another process; retrying"; continue 2; fi
        echo "[vllm] died; tail of log:"; tail -30 "$VLLM_LOG"; return 1
      fi
      if [ $t -ge ${VLLM_TIMEOUT:-900} ]; then echo "[vllm] timeout waiting for /health"; tail -30 "$VLLM_LOG"; return 1; fi
    done
    root=$(_vllm_root)
    if [ -n "$root" ] && [ "$(readlink -f "$root")" = "$(readlink -f "$MODEL")" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "[vllm] ready after ${t}s (serving $root)"; return 0
    fi
    echo "[vllm] :$PORT answers with another server (root=${root:-?}), not ours; stopping ours and retrying (attempt $attempt)"
    kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null
  done
  return 1
}
vllm_stop() {
  if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then kill "$VLLM_PID"; wait "$VLLM_PID" 2>/dev/null; echo "[vllm] stopped"; fi
}
