#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

MODELS=(
    "test_llama_3_1_8b:llama_3_1_8b"
    "test_qwen_3_8b:qwen_3_8b"
    # "test_llama_3_1_70b_tp:llama_3_1_70b"
)

for model_entry in "${MODELS[@]}"; do
    test_fn="${model_entry%%:*}"
    model_name="${model_entry##*:}"
    for bs in 1 8 16 32; do
        for seq in 1024 512 128; do
            out_dir="results/${model_name}_bs${bs}_seq${seq}"
            mkdir -p "$out_dir"
            TT_METAL_DEVICE_PROFILER=1 python3 -m tracy -v -r -p \
                -o "${out_dir}/tracy" \
                --tracy-tools-folder "$(pwd)/third_party/tt-mlir/install/bin" \
                -m "pytest tests/benchmark/test_llms.py::${test_fn} \
                    --prefill-only \
                    --batch-size $bs \
                    --input-sequence-length $seq \
                    --output-file ${out_dir}/metrics.json \
                    --optimization-level 0"
        done
    done
done
