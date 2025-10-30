# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

LOGGER_LEVEL=DEBUG vllm serve Qwen/Qwen3-Embedding-4B \
    --max-model-len 16384 \
    --max-num-batched-tokens 16384 \
    --max-num-seqs 1 \
    --no-enable-prefix-caching \
    --additional-config "{\"enable_const_eval\": \"False\"}"
