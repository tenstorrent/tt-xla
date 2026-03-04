# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --max-model-len 1024 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 1 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.1 \
    --additional-config "{\"enable_const_eval\": \"False\", \"min_context_len\": 32}"
