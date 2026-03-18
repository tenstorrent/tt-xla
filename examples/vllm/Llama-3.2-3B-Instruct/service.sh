# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --max-model-len 2048 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 1 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.05 \
    --additional-config "{\"enable_const_eval\": \"False\", \"min_context_len\": 32}"
