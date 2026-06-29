#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Serve Qwen3-0.6B from previously emitted TTNN codegen instead of compiling.
#
#   1. emit once (offline) to populate the codegen dir:
#        python qwen.py --emit [dir]      # default ./qwen_codegen
#   2. start the server pointing at that same dir:
#        ./service.sh [dir]               # default ./qwen_codegen
#   3. chat with it:
#        python client.py
#
# TTXLA_CODEGEN_LOAD_DIR makes the TT plugin match each graph by StableHLO hash
# against the saved subdirs and run the (optionally edited) main.py, skipping
# SHLO->TTIR->TTNN compilation. The server's "spawn" workers inherit this env
# var. The flags below MUST match the emit run (qwen.py) so the hashes line up.
#
# NOTE: in this build the OpenAI server currently 500s on every request due to a
# prometheus-fastapi-instrumentator==8.0.0 incompatibility (registered
# unconditionally in vllm/entrypoints/serve). Until that is resolved, use the
# offline loader instead: python qwen.py --load [dir]

set -euo pipefail

CODEGEN_DIR="${1:-qwen_codegen}"
if [ ! -d "$CODEGEN_DIR" ]; then
    echo "codegen dir '$CODEGEN_DIR' does not exist -- run 'python qwen.py --emit $CODEGEN_DIR' first" >&2
    exit 1
fi
export TTXLA_CODEGEN_LOAD_DIR="$(cd "$CODEGEN_DIR" && pwd)"
echo "serving from codegen dir: $TTXLA_CODEGEN_LOAD_DIR"

vllm serve Qwen/Qwen3-0.6B \
    --max-model-len 512 \
    --max-num-batched-tokens 512 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.02 \
    --additional-config "{\"enable_const_eval\": \"False\", \"min_context_len\": 32}"
