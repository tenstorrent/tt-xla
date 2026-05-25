# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Launches a TT-backed vLLM server for Llama-3.2-3B.
#
# Usage:
#   bash service.sh                  # default BATCH_SIZE=32
#   BATCH_SIZE=1  bash service.sh    # single-user
#   BATCH_SIZE=8  bash service.sh    # 8-user
#   BATCH_SIZE=32 bash service.sh    # 32-user (mirrors the benchmark setup)
#
# Companion to the slow-prefill investigation tracked in tt-xla #4880 (and follow-up
# perf bug). max_model_len=128 keeps runs short for iteration.
#
# Expects the standard tt-xla env to be active and an HF token / cached model for
# `meta-llama/Llama-3.2-3B`. Also recommended on Blackhole p150:
#   TT_MESH_GRAPH_DESC_PATH=$TT_METAL_HOME/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto
#   TT_VISIBLE_DEVICES=0

set -euo pipefail

MODEL="${MODEL:-meta-llama/Llama-3.2-3B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-128}"
BATCH_SIZE="${BATCH_SIZE:-32}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.05}"
# Mirror the per-batch-size memory tuning used in tests/benchmark/test_vllm_benchmarks.py
# (batch=32 needs slightly less per-rank memory because there are more concurrent KV
# cache pages; values below match _config / SINGLE_DEVICE_CONFIGS for llama-3.2-3b{,-batch32}).
if [ "$BATCH_SIZE" -eq 32 ] && [ "$GPU_MEMORY_UTILIZATION" = "0.05" ]; then
    GPU_MEMORY_UTILIZATION=0.037
fi

MAX_NUM_BATCHED_TOKENS=$((BATCH_SIZE * MAX_MODEL_LEN))

echo "Launching vLLM with:"
echo "  model                   = $MODEL"
echo "  max_model_len           = $MAX_MODEL_LEN"
echo "  max_num_seqs            = $BATCH_SIZE"
echo "  max_num_batched_tokens  = $MAX_NUM_BATCHED_TOKENS"
echo "  gpu_memory_utilization  = $GPU_MEMORY_UTILIZATION"
echo

# `enable_trace=True` matches the benchmark; trace replay is what amortizes the
# dispatched-ops cost into a single device call after the first iteration.
exec vllm serve "$MODEL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$BATCH_SIZE" \
    --no-enable-prefix-caching \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --additional-config "{\"enable_trace\": true, \"cpu_sampling\": false}"
