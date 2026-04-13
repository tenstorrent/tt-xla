# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# OPT-125M vLLM server with metal trace support.
# Toggle trace mode: ENABLE_TRACE=true ./service.sh  (default: false)

ENABLE_TRACE="${ENABLE_TRACE:-false}"

echo "Starting OPT-125M server (enable_trace=${ENABLE_TRACE}, greedy, device sampling)"

vllm serve facebook/opt-125m \
    --max-model-len 512 \
    --max-num-batched-tokens 512 \
    --max-num-seqs 1 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.05 \
    --additional-config "{\"enable_const_eval\": false, \"min_context_len\": 32, \"cpu_sampling\": false, \"enable_trace\": ${ENABLE_TRACE}}"
