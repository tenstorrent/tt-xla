# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Llama-3.2-1B vLLM server with metal trace support.
# Toggle trace mode: ENABLE_TRACE=true ./service.sh  (default: false)
# NOTE: trace is disabled by default due to RoPE from_device issue (see trace_investigation_2026-04-12.md)

ENABLE_TRACE="${ENABLE_TRACE:-false}"

echo "Starting Llama-3.2-1B server (enable_trace=${ENABLE_TRACE}, greedy, device sampling)"

vllm serve meta-llama/Llama-3.2-1B \
    --max-model-len 512 \
    --max-num-batched-tokens 512 \
    --max-num-seqs 1 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.1 \
    --additional-config "{\"enable_const_eval\": false, \"min_context_len\": 32, \"cpu_sampling\": false, \"enable_trace\": ${ENABLE_TRACE}}"
