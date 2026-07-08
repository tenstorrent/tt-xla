# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Serve Devstral-2-123B-Instruct on the 32-chip Blackhole galaxy in a 4x8
# DP+TP layout (mesh_shape=[4, 8] -> 4 data-parallel replicas x 8-way tensor
# parallel). Mirrors the validated perf benchmark
# (tests/benchmark/test_vllm_benchmarks.py::...[devstral-123b-galaxy-tp]).
#
# Prereqs:
#   - 32-chip BH galaxy; build against the custom tt-mlir/tt-metal galaxy
#     branch (TT_MLIR_VERSION=ssalice/devstral-wip-06252026-mlir).
#   - fp8 checkpoint: the plugin's fp8->bf16 dequant hook installs itself.
#
# Then chat:  python examples/vllm/Devstral-2-123B-Instruct-2512/client.py
#
# NOTE: this is the DP+TP *batched-throughput* config. max_num_seqs=4 (= dp_size,
# the minimum a DP mesh allows); to exercise all 4 replicas send up to 4
# concurrent requests. The throughput benchmark validated max_model_len=128;
# 1024 here is for interactive use and fits the KV budget because max_num_seqs
# is far smaller than the benchmark's 128 (KV scales with seqs x len). If the
# galaxy build fails to compile at this length, drop --max-model-len to 128.

set -euo pipefail

export TT_RUNTIME_USING_BH_GALAXY=1

vllm serve mistralai/Devstral-2-123B-Instruct-2512 \
    --max-model-len 1024 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.3 \
    --additional-config "{\"mesh_shape\": [4, 8], \"enable_tensor_parallel\": true, \"enable_data_parallel\": true, \"shard_weights_on_batch_axis\": false, \"experimental_weight_dtype\": \"bfp_bf8\", \"min_context_len\": 32}"
