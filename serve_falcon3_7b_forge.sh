#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Falcon3-7B-Instruct forge LLM served by STOCK `vllm serve`, from tt-xla only.
#
# Purpose: reproduce the tt-inference-server serving config with NO
# tt-inference-server / tt-media-server in the loop, so a hang can be blamed on
# (or cleared of) the tt-xla vLLM plugin rather than tt-media-server's driver.
# Pairs with ./run_falcon3_7b_evals.sh, which drives the same two evals
# (ifeval + gpqa_diamond_generative_n_shot) that `run_evals_forge.sh` runs.
#
#   cd ~/tt-xla && source venv/activate && ./serve_falcon3_7b_forge.sh
#
# Config parity is exact against the observed live run
# (~/tt-inference-server/falcon_single_layer.log, "Device 0: additional_config=…"
# + "Initializing a V1 LLM engine"), which mirrors the forge P150 spec in
# workflows/model_specs/dev/cnn.yaml: b32, 32768 ctx, chunk 1024, gmu 0.35,
# bfp8 weights+KV, opt=1, device sampling, trace, b1-prefill. Full model depth
# by default; set NUM_HIDDEN_LAYERS=1 for the single-layer debug hack.
#
# WHAT IS DELIBERATELY *NOT* IDENTICAL (and why it matters for #4521-class hangs)
#   - Driver. tt-media-server runs AsyncLLMEngine behind its own
#     `device_worker_dynamic_batch` (get_many(32) -> burst of generate() ->
#     keep pulling), which is what reliably tripped the old scheduler deadlock.
#     `vllm serve` uses vLLM-native admission. Removing that layer is the point
#     of this script, but it also means a hang that ONLY tt-media-server's
#     driver can trigger will not show up here. A clean run is evidence, not
#     proof.
#   - Sampling seed. tt-media-server force-drops per-request seeds (#4338 /
#     tt-xla#4539); vLLM honors whatever lm-eval sends. Both paths still run
#     greedy (lm-eval sends temperature=0), so this is a speed/path difference,
#     not an expected accuracy difference. See run_falcon3_7b_evals.sh's
#     EVAL_GEN_SEED knob.
#   - The scheduler IS the same: vllm_tt's platform.py sets
#     scheduler_cls="vllm_tt.scheduler.AscendScheduler" for TT regardless of
#     entry point, so AscendScheduler is exercised either way.
#
# Env knobs (all overridable; defaults reproduce the observed run):
#   PORT, DEVICE_IDS, MODEL_REPO, NUM_HIDDEN_LAYERS, MAX_MODEL_LENGTH,
#   MAX_NUM_SEQS, GPU_MEMORY_UTILIZATION, PREFILL_CHUNK_SIZE, MIN_NUM_SEQS,
#   PREFILL_BATCH_THRESHOLD, OPTIMIZATION_LEVEL, CPU_SAMPLING, ENABLE_TRACE,
#   KV_CACHE_DTYPE, WEIGHT_DTYPE, MIN_CONTEXT_LEN, TT_KV_POOL_GB,
#   MATH_FIDELITY, FP32_DEST_ACC_EN, API_KEY, WARMUP, TT_METAL_HOME,
#   DISABLE_PREFIX_CACHING (debug opt-in, 0 by default: 1 passes
#   --no-enable-prefix-caching -- vLLM's cross-request KV-reuse feature, NOT the
#   "cached-prefix"/chunked-SDPA compile axis controlled by PREFILL_CHUNK_SIZE)
set -eo pipefail  # NOT -u: venv/activate references vars unset until sourced

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- tt-metal runtime env -----------------------------------------------------
# TT_METAL_HOME here is only a writable JIT/kernel cache root (the real tt-metal
# lives under third_party/); tt-media-server points it at "$(pwd)/tt-metal" and
# then sets TT_METAL_CACHE=$TT_METAL_HOME/built/<device_id>. Mirrored so cache
# layout matches. Point it at an existing dir to reuse a warm compile cache.
export TT_METAL_HOME="${TT_METAL_HOME:-$ROOT/tt-metal}"
export TT_MESH_GRAPH_DESC_PATH="${TT_MESH_GRAPH_DESC_PATH:-$ROOT/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto}"

export DEVICE_IDS="${DEVICE_IDS:-0}"
export TT_VISIBLE_DEVICES="$DEVICE_IDS"
export TT_METAL_CACHE="$TT_METAL_HOME/built/${DEVICE_IDS//,/_}"
mkdir -p "$TT_METAL_CACHE"

# tt-media-server's setup_cpu_threading_limits(cpu_threads="2", torch=1).
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

# --- model / engine config ----------------------------------------------------
MODEL_REPO="${MODEL_REPO:-tiiuae/Falcon3-7B-Instruct}"
PORT="${PORT:-8019}"
HOST="${HOST:-0.0.0.0}"

MAX_MODEL_LENGTH="${MAX_MODEL_LENGTH:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE-1024}"
MIN_NUM_SEQS="${MIN_NUM_SEQS-1}"
PREFILL_BATCH_THRESHOLD="${PREFILL_BATCH_THRESHOLD-16}"
OPTIMIZATION_LEVEL="${OPTIMIZATION_LEVEL:-1}"
CPU_SAMPLING="${CPU_SAMPLING:-false}"
ENABLE_TRACE="${ENABLE_TRACE:-true}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-bfp_bf8}"
WEIGHT_DTYPE="${WEIGHT_DTYPE:-bfp_bf8}"
MIN_CONTEXT_LEN="${MIN_CONTEXT_LEN:-128}"
# Full model by default. Set NUM_HIDDEN_LAYERS=1 for the single-layer debug hack.
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS-}"
# Debug opt-in: disable vLLM's cross-request prefix caching entirely (default
# on in production). Not a compile-time lever -- purely to see if it affects
# the hang.
DISABLE_PREFIX_CACHING="${DISABLE_PREFIX_CACHING:-0}"
export TT_KV_POOL_GB="${TT_KV_POOL_GB:-32}"

# Same derivation as tt-media-server's VLLMSettings: with chunked prefill the
# budget is batch*chunk floored at max_model_length (clears vLLM's
# max_num_batched_tokens >= max_model_len check); otherwise batch*full-context.
if [ -n "$PREFILL_CHUNK_SIZE" ] && [ "$PREFILL_CHUNK_SIZE" -gt 0 ] 2>/dev/null; then
  MAX_NUM_BATCHED_TOKENS=$(( MAX_NUM_SEQS * PREFILL_CHUNK_SIZE ))
  [ "$MAX_NUM_BATCHED_TOKENS" -lt "$MAX_MODEL_LENGTH" ] && MAX_NUM_BATCHED_TOKENS="$MAX_MODEL_LENGTH"
else
  MAX_NUM_BATCHED_TOKENS=$(( MAX_MODEL_LENGTH * MAX_NUM_SEQS ))
fi

# additional_config, key-for-key and in the same insertion order as
# VLLMForgeRunner.warmup() builds it (vLLM hashes this dict; keep it stable).
ADDITIONAL_CONFIG=$(
  MIN_CONTEXT_LEN="$MIN_CONTEXT_LEN" WEIGHT_DTYPE="$WEIGHT_DTYPE" \
  KV_CACHE_DTYPE="$KV_CACHE_DTYPE" CPU_SAMPLING="$CPU_SAMPLING" \
  OPTIMIZATION_LEVEL="$OPTIMIZATION_LEVEL" ENABLE_TRACE="$ENABLE_TRACE" \
  PREFILL_CHUNK_SIZE="$PREFILL_CHUNK_SIZE" FP32_DEST_ACC_EN="$FP32_DEST_ACC_EN" \
  MATH_FIDELITY="$MATH_FIDELITY" MIN_NUM_SEQS="$MIN_NUM_SEQS" \
  PREFILL_BATCH_THRESHOLD="$PREFILL_BATCH_THRESHOLD" \
  NUM_HIDDEN_LAYERS="$NUM_HIDDEN_LAYERS" \
  python3 - <<'PY'
import json, os

def env(name):
    v = os.environ.get(name, "")
    return v.strip()

cfg = {
    "enable_const_eval": True,
    "min_context_len": int(env("MIN_CONTEXT_LEN")),
    "experimental_weight_dtype": env("WEIGHT_DTYPE"),
    "experimental_kv_cache_dtype": env("KV_CACHE_DTYPE"),
    "cpu_sampling": env("CPU_SAMPLING").lower() == "true",
    "optimization_level": int(env("OPTIMIZATION_LEVEL")),
    "enable_trace": env("ENABLE_TRACE").lower() == "true",
}
# Optional keys: only passed when set, matching the runner (an unset key keeps
# the plugin default; an empty-string key would not).
if env("PREFILL_CHUNK_SIZE"):
    cfg["prefill_chunk_size"] = int(env("PREFILL_CHUNK_SIZE"))
if env("FP32_DEST_ACC_EN"):
    cfg["fp32_dest_acc_en"] = env("FP32_DEST_ACC_EN").lower() == "true"
if env("MATH_FIDELITY"):
    cfg["math_fidelity"] = env("MATH_FIDELITY")
if env("MIN_NUM_SEQS"):
    cfg["min_num_seqs"] = int(env("MIN_NUM_SEQS"))
if env("PREFILL_BATCH_THRESHOLD"):
    cfg["prefill_batch_threshold"] = int(env("PREFILL_BATCH_THRESHOLD"))
if env("NUM_HIDDEN_LAYERS"):
    cfg["num_hidden_layers"] = int(env("NUM_HIDDEN_LAYERS"))
print(json.dumps(cfg))
PY
)

LOG_DIR="${LOG_DIR:-$ROOT}"
LOG="${LOG:-$LOG_DIR/serve_falcon3_7b_forge_dev${DEVICE_IDS//,/_}_p${PORT}_$(date +%Y%m%d_%H%M%S).log}"

echo "=============================================================="
echo " Falcon3-7B-Instruct  |  stock vllm serve  |  tt-xla standalone"
echo "   model      : $MODEL_REPO"
echo "   device     : TT_VISIBLE_DEVICES=$TT_VISIBLE_DEVICES   port=$PORT"
echo "   layers     : ${NUM_HIDDEN_LAYERS:-<full model>}"
echo "   b/ctx/chunk: $MAX_NUM_SEQS / $MAX_MODEL_LENGTH / $PREFILL_CHUNK_SIZE"
echo "   batched tok: $MAX_NUM_BATCHED_TOKENS   gmu=$GPU_MEMORY_UTILIZATION"
echo "   addl config: $ADDITIONAL_CONFIG"
[ "$DISABLE_PREFIX_CACHING" = "1" ] && echo "   prefix cache: DISABLED (--no-enable-prefix-caching)"
echo "   TT_METAL_CACHE=$TT_METAL_CACHE"
echo "   log        : $LOG"
echo "=============================================================="

# Mirror VLLMForgeRunner.warmup()'s post-load warmup generate. `vllm serve` has
# no warmup of its own, so without this the FIRST eval request pays the one-time
# MLIR/kernel compile -- exactly the "false hang" the #4521 handoff warns about.
# Fires in the background once the server reports ready.
if [ "${WARMUP:-1}" = "1" ]; then
  (
    for _ in $(seq 1 720); do
      curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && break
      sleep 5
    done
    echo "[warmup] server up; sending warmup completion (this compiles the graphs)..."
    curl -s -X POST "http://127.0.0.1:${PORT}/v1/completions" \
      -H "Content-Type: application/json" \
      ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
      -d "{\"model\":\"${MODEL_REPO}\",\"prompt\":\"Hello, it's me\",\"max_tokens\":10,\"temperature\":0}" \
      >/dev/null 2>&1 \
      && echo "[warmup] WARMUP COMPLETE — server is warm, safe to start evals" \
      || echo "[warmup] warmup request FAILED (check the log above)"
  ) &
fi

# --no-enable-chunked-prefill matches the AsyncEngineArgs(enable_chunked_prefill=
# False) the runner passes. vllm_tt's platform flips it back on internally (the
# live run logs enable_chunked_prefill=True) and re-derives batch*chunk itself.
vllm serve "$MODEL_REPO" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LENGTH" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --no-enable-chunked-prefill \
  --additional-config "$ADDITIONAL_CONFIG" \
  ${API_KEY:+--api-key "$API_KEY"} \
  $([ "$DISABLE_PREFIX_CACHING" = "1" ] && echo --no-enable-prefix-caching) \
  "$@" \
  |& tee "$LOG"
