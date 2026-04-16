#!/usr/bin/env bash
# Llama-3.1-8B sampling performance profile.
# Runs full benchmarks + short tracy captures for dispatch overhead analysis.
#
# Full benchmarks: accurate tok/s numbers
# Tracy runs: 4 tokens for short traces, use to compare host dispatch timelines
#
# Usage: bash run_sampling_perf_profile.sh

set -e

# echo "===== [1/4] Full benchmark: greedy ====="
# # tt-smi -r 0
# # sleep 1
# pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-greedy-device]" |& tee llama31_8b_greedy_full.log

# echo ""
# echo "===== [2/4] Full benchmark: non-greedy (no penalty) ====="
# # tt-smi -r 0
# # sleep 1
# TT_USE_TTNN_SAMPLING=1 pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-nongreedy-nopenalty-device]" |& tee llama31_8b_nongreedy_nopenalty_full.log

echo ""
echo "===== [3/4] Tracy: greedy (4 tokens) ====="
# tt-smi -r 0
# sleep 1
VLLM_ENABLE_V1_MULTIPROCESSING=0 python -m tracy -p -r -o tracy_greedy_4tok venv/bin/pytest -svv --max-output-tokens 4 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-greedy-device]" |& tee llama31_8b_greedy_tracy.log

echo ""
echo "===== [4/4] Tracy: non-greedy (4 tokens) ====="
# tt-smi -r 0
# sleep 1
VLLM_ENABLE_V1_MULTIPROCESSING=0 TT_USE_TTNN_SAMPLING=1 python -m tracy -p -r -o tracy_nongreedy_4tok venv/bin/pytest -svv --max-output-tokens 4 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-nongreedy-nopenalty-device]" |& tee llama31_8b_nongreedy_tracy.log

echo ""
echo "===== Done ====="
echo "Benchmark logs:   llama31_8b_greedy_full.log"
echo "                  llama31_8b_nongreedy_nopenalty_full.log"
echo "Tracy traces:     tracy_greedy_4tok.tracy"
echo "                  tracy_nongreedy_4tok.tracy"
