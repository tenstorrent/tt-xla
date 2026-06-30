#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Standalone `vllm serve` for Qwen3-8B on the forge plugin, used to reproduce
# the EngineCore host-memory leak (see leak_probe.py + README.md). Mirrors the
# tt-inference-server forge release serving config but with no tt-media-server
# wrapper, so the leak is isolated to the plugin / vLLM EngineCore.
#
# Run from the tt-xla venv. Standalone needs a single chip pinned plus, on a
# CUSTOM-cluster box, the mesh descriptor:
#   cd ~/tt-xla && source venv/activate && TT_VISIBLE_DEVICES=0 ./service.sh
#
# Defaults = the RECOMMENDED fast leak repro: single layer + 2k context + no
# chunked prefill + b32. It compiles in ~2 min, floods in ~100s, and still shows
# the leak (~3 KB/tok, sticky). Override any knob to test what the leak needs or
# to run the full production config, e.g.:
#   NUM_HIDDEN_LAYERS=full MAX_MODEL_LEN=40960 PREFILL_CHUNK_SIZE=2048 ./service.sh
#
# Env knobs:
#   NUM_HIDDEN_LAYERS=1        1 = single layer (default); "full" = all layers
#   MAX_MODEL_LEN=2048         context (40960 for the production long-context cfg)
#   MAX_NUM_SEQS=32            batch (b32)
#   PREFILL_CHUNK_SIZE=0       0 = no chunked prefill + no b1-prefill (default);
#                              >0 = chunked-prefill chunk size + b1-prefill
#   OPT_LEVEL=1   ENABLE_TRACE=true   PORT=8000
set -e

export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-0}
export TT_MESH_GRAPH_DESC_PATH=${TT_MESH_GRAPH_DESC_PATH:-/home/kmabee/tt-xla/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto}

# Exported so the additional-config python heredoc below sees them too.
export MODEL=${MODEL:-Qwen/Qwen3-8B}
export NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS:-1}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-2048}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
export PREFILL_CHUNK_SIZE=${PREFILL_CHUNK_SIZE:-0}
export OPT_LEVEL=${OPT_LEVEL:-1}
export ENABLE_TRACE=${ENABLE_TRACE:-true}
PORT=${PORT:-8000}

# Build the tt additional-config from the env knobs (a scalar number value maps
# to the matching TTConfig field). prefill_chunk_size>0 also enables b1-prefill.
ADDITIONAL_CONFIG=$(python3 - <<'PY'
import json, os
cfg = {
    "enable_const_eval": True,
    "min_context_len": 128,
    "experimental_weight_dtype": "bfp_bf8",
    "experimental_kv_cache_dtype": "bfp_bf8",
    "cpu_sampling": False,
    "optimization_level": int(os.environ.get("OPT_LEVEL", "1")),
    "enable_trace": os.environ.get("ENABLE_TRACE", "true").lower() == "true",
    "fp32_dest_acc_en": False,
}
nhl = os.environ.get("NUM_HIDDEN_LAYERS", "1").strip()
if nhl and nhl.lower() not in ("full", "0"):  # "full"/0 => all layers
    cfg["num_hidden_layers"] = int(nhl)
chunk = int(os.environ.get("PREFILL_CHUNK_SIZE", "0") or 0)
if chunk > 0:
    cfg["prefill_chunk_size"] = chunk
    cfg["min_num_seqs"] = 1          # b1-prefill (compile [1,n] alongside [32,n])
    cfg["prefill_batch_threshold"] = 16
print(json.dumps(cfg))
PY
)

# Chunked prefill: enabled iff PREFILL_CHUNK_SIZE>0; otherwise plain b32.
# With chunked prefill ON the plugin right-sizes max_num_batched_tokens; with it
# OFF vLLM requires max_num_batched_tokens >= max_model_len * max_num_seqs, so
# default to that (override with MAX_NUM_BATCHED_TOKENS).
CHUNK_FLAGS=()
if [ "${PREFILL_CHUNK_SIZE:-0}" -gt 0 ] 2>/dev/null; then
    CHUNK_FLAGS+=(--enable-chunked-prefill)
    MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$MAX_MODEL_LEN}
else
    CHUNK_FLAGS+=(--no-enable-chunked-prefill)
    MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$((MAX_MODEL_LEN * MAX_NUM_SEQS))}
fi

echo "Starting vllm serve: MODEL=$MODEL MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS PREFILL_CHUNK_SIZE=$PREFILL_CHUNK_SIZE NUM_HIDDEN_LAYERS=${NUM_HIDDEN_LAYERS:-full} PORT=$PORT"
echo "additional-config: $ADDITIONAL_CONFIG"

vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --block-size 64 \
    --gpu-memory-utilization 0.30 \
    --no-enable-prefix-caching \
    "${CHUNK_FLAGS[@]}" \
    --additional-config "$ADDITIONAL_CONFIG"
