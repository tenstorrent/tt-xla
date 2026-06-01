#!/usr/bin/env bash
# Sweep prefill benchmarks for llama 3.1 8b (single chip) and 70b (tensor parallel).
# Loops over batch size, sequence length, and optimization level.
set -uo pipefail

cd "$(dirname "$0")"

BATCH_SIZES=(1) # 8 16 32)
SEQ_LENS=(1024) # 512 128)
OPT_LEVELS=(0 1)

# test name : short tag : num_layers ("" = full model, only tp gets 1 layer)
TESTS=(
    #"test_llama_3_1_8b:llama_3_1_8b:"
    #"test_llama_3_1_70b_tp:70b:1"
    #"test_qwen_3_8b:qwen_3_8b:"
    "test_llama_3_2_1b:llama_3_2_1b:"
    "test_gemma_1_1_2b:gemma_1_1_2b:"
    "test_phi1:phi1:"
)

RESULTS_DIR="prefill_sweep_results"
mkdir -p "$RESULTS_DIR"

for entry in "${TESTS[@]}"; do
    IFS=':' read -r test_name tag num_layers <<< "${entry}"
    for opt in "${OPT_LEVELS[@]}"; do
        for sl in "${SEQ_LENS[@]}"; do
            for bs in "${BATCH_SIZES[@]}"; do
                out="${RESULTS_DIR}/${tag}_bs${bs}_sl${sl}_opt${opt}.json"
                layers_arg=()
                [[ -n "${num_layers}" ]] && layers_arg=(--num-layers "${num_layers}")
                echo "=== ${test_name} | bs=${bs} sl=${sl} opt=${opt} layers=${num_layers:-full} -> ${out} ==="
                pytest "tests/benchmark/test_llms.py::${test_name}" \
                    "${layers_arg[@]}" \
                    --batch-size "${bs}" \
                    --prefill-only \
                    --input-sequence-length "${sl}" \
                    --optimization-level "${opt}" \
                    --output-file "${out}" -svv
            done
        done
    done
done
