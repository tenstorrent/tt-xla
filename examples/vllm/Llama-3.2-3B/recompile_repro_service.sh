#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Repro for the prefill last-token-select runtime recompile (regression from #4789).
#
# Serves Llama-3.2-3B on the **fused / device-sampling** path (cpu_sampling=False),
# the affected configuration. `_precompile_decode_postprocess` only precompiles the
# decode shape (num_tokens==1), but execute_model selects the prefill last token with
# an eager gather OUTSIDE decode_postprocess for num_tokens>1 — a distinct graph per
# prefill token bucket that is JIT-compiled at runtime on first use.
#
# VLLM_XLA_CHECK_RECOMPILATION=1 turns on the built-in detector: after the engine's
# graph set stabilizes it raises "Recompilation after warm up is detected" the first
# time a new graph compiles. With this repro, that fires on the first request whose
# prompt length lands in a NEW prefill bucket (see recompile_repro_client.py).
#
# Set TT_DISABLE_PREFILL_SELECT_PRECOMPILE=1 to observe the STOCK (pre-fix) behavior;
# unset (default) exercises the fix (no runtime recompile).
set -euo pipefail
# Repo root, derived from this script's location (examples/vllm/Llama-3.2-3B/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-0}
# Single-chip Blackhole (P150) mesh descriptor; override for other hardware.
export TT_MESH_GRAPH_DESC_PATH=${TT_MESH_GRAPH_DESC_PATH:-$ROOT/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto}
export VLLM_XLA_CHECK_RECOMPILATION=${VLLM_XLA_CHECK_RECOMPILATION:-1}

vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --max-model-len 4096 \
    --max-num-batched-tokens 65536 \
    --max-num-seqs 1 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.40 \
    --additional-config "{\"cpu_sampling\": false, \"experimental_kv_cache_dtype\": \"none\", \"experimental_weight_dtype\": \"bfp_bf8\", \"optimization_level\": 1, \"enable_const_eval\": true, \"min_context_len\": 128}"
