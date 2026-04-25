# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

vllm serve HuggingFaceTB/SmolLM3-3B \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 32 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.90 \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --additional-config "{\"enable_const_eval\": \"False\", \"min_context_len\": 32}"
