#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Launch vLLM server for bleed reproduction.
#
# Usage:
#   ./server_bleed_repro.sh                    # with prefix caching (bleeds)
#   ./server_bleed_repro.sh --no-prefix-cache  # workaround (no bleed)
#
# Then run client_bleed_repro.py in a separate terminal.

set -e

MODEL="${MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
PORT="${PORT:-8199}"

EXTRA_ARGS=""
if [[ "$1" == "--no-prefix-cache" ]]; then
    echo ">>> Prefix caching DISABLED (workaround mode)"
    EXTRA_ARGS="--no-enable-prefix-caching"
else
    echo ">>> Prefix caching ENABLED (default — may bleed on round 2+)"
fi

echo ">>> Model: $MODEL"
echo ">>> Port: $PORT"
echo ">>> Starting server..."

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --max-model-len 512 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.05 \
    --port "$PORT" \
    --additional-config '{"enable_const_eval": false, "min_context_len": 32, "cpu_sampling": true}' \
    $EXTRA_ARGS
