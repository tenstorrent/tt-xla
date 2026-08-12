# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Serve Qwen3-32B on the 32-chip Blackhole galaxy in an 8x4 DP+TP layout
# (mesh_shape=[8, 4] -> 8 data-parallel replicas x 4-way tensor parallel).
# Mirrors the validated perf benchmark
# (tests/benchmark/test_vllm_benchmarks.py::...[qwen3-32b-galaxy-tp]).
#
# Prereqs:
#   - 32-chip BH galaxy; build against the custom tt-mlir/tt-metal galaxy
#     branch (TT_MLIR_VERSION=ssalice/devstral-wip-06252026-mlir).
#
# Then chat:  python examples/vllm/Qwen3-32B/client.py
#
# NOTE: this is the DP+TP *batched-throughput* config. max_num_seqs=8 (= dp_size,
# the minimum an 8-way DP mesh allows); to exercise all 8 replicas send up to 8
# concurrent requests. The throughput benchmark validated max_model_len=128;
# 1024 here is for interactive use and fits the KV budget because max_num_seqs
# is far smaller than the benchmark's 256 (KV scales with seqs x len). If the
# galaxy build fails to compile at this length, drop --max-model-len to 128.

set -euo pipefail

export TT_RUNTIME_USING_BH_GALAXY=1

vllm serve Qwen/Qwen3-32B \
    --max-model-len 1024 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 8 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.45 \
    --additional-config "{\"mesh_shape\": [8, 4], \"enable_tensor_parallel\": true, \"enable_data_parallel\": true, \"shard_weights_on_batch_axis\": false, \"experimental_weight_dtype\": \"bfp_bf8\", \"enable_const_eval\": true, \"min_context_len\": 32}"
