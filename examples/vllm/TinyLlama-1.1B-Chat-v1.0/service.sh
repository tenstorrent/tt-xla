# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

TT_RUNTIME_ENABLE_PROGRAM_CACHE=1 LOGGER_LEVEL=DEBUG vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --max-model-len 128 \
    --max-num-batched-tokens 128 \
    --max-num-seqs 1 \
    --no-enable-prefix-caching \
    --additional-config "{\"enable_const_eval\": \"False\", \"min_context_len\": 32}"
