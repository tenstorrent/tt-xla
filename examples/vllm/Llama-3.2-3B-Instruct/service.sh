# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

BATCH_SIZE=${1:-1}

# max_model_len * max_num_seqs must fit in max_num_batched_tokens, and the
# resulting tensor shapes must fit in DRAM. Scale down context length as
# batch size increases to stay within hardware limits.
MAX_MODEL_LEN=$((2048 / BATCH_SIZE))
MAX_BATCHED_TOKENS=$((MAX_MODEL_LEN * BATCH_SIZE))

echo "Starting Llama-3.2-3B-Instruct with batch_size=$BATCH_SIZE, max_model_len=$MAX_MODEL_LEN"

vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --max-num-seqs "$BATCH_SIZE" \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.05 \
    --additional-config "{\"enable_const_eval\": \"False\", \"min_context_len\": 32}"
