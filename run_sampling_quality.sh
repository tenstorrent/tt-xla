#!/usr/bin/env bash
# Sampling quality tests: greedy and non-greedy (temp=1.0) device sampling
# for OPT-125M, Llama-3.2-1B, Llama-3.1-8B.
#
# Usage:
#   bash run_sampling_quality.sh                     # device sampling (default)
#   TT_USE_TTNN_SAMPLING=1 bash run_sampling_quality.sh  # with ttnn.sampling op
#
# Inspect the "  [0] prompt → output" lines to check output quality.

# Note: no set -e — we want all tests to run even if individual ones fail.
# Also: run with "bash run_sampling_quality.sh", not "source", to avoid
# closing your terminal on failure.

# LOG_DIR="sampling_quality_mlir_update_const_eval_bfp8_$(date +%Y%m%d_%H%M%S)"
# LOG_DIR="sampling_quality_mlir_update_const_eval_bfp8_no_mlir_uplift_$(date +%Y%m%d_%H%M%S)"
# LOG_DIR="sampling_quality_sort_replace_topk_fix_gumbel_hack_$(date +%Y%m%d_%H%M%S)"
# LOG_DIR="sampling_quality_sort_replace_topk_gather_fix_threshold_sampling_cleanup_tt_sampling_cleanup_opt_level_1_fix_pad32_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="sampling_quality_sort_sort_gather_no_sampling_opt0_$(date +%Y%m%d_%H%M%S)"


LOG_DIR="sampling_quality_baseline_revert_rope_dim2_fix_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="sampling_quality_baseline_batch_2_tt_sampling_opt_level_1_$(date +%Y%m%d_%H%M%S)"

# LOG_DIR="sampling_quality_baseline"

mkdir -p "$LOG_DIR"

run() {
    local id="$1"
    echo ""
    echo "===== $id ====="
    pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_sampling_quality[$id]" \
        |& tee "$LOG_DIR/${id}.log"
}

# run "llama3.2-1b-greedy-device"
# run "llama3.2-1b-nongreedy-device"
# run "llama3.2-1b-greedy-cpu"
# run "llama3.2-1b-nongreedy-cpu"

# run "llama3.1-8b-greedy-device"
# run "llama3.1-8b-nongreedy-device"
# run "llama3.1-8b-greedy-cpu"
# run "llama3.1-8b-nongreedy-cpu"

run "llama3.2-1b-b2-greedy-device"
run "llama3.2-1b-b2-nongreedy-device"
run "llama3.2-1b-b2-greedy-cpu"
run "llama3.2-1b-b2-nongreedy-cpu"

run "llama3.1-8b-b2-greedy-device"
run "llama3.1-8b-b2-nongreedy-device"
run "llama3.1-8b-b2-greedy-cpu"
run "llama3.1-8b-b2-nongreedy-cpu"

# Run sampler throughtput tests
python perf_debug/test_sampler_throughput.py |& tee "$LOG_DIR/sampler_throughput_non_greedy.log"
python perf_debug/test_sampler_throughput.py --top-k 3 |& tee "$LOG_DIR/sampler_throughput_non_greedy_topk.log"

echo ""
echo "===== Output Summary ====="
grep -hE "^\s*\[[0-9]+\]" "$LOG_DIR"/*.log
echo ""
echo "Logs in: $LOG_DIR"


# TRACY_PROFILING_ACTIVE=1 python -m tracy -p -r -o tracy_sampler_nongreedy_tt_sampling -m perf_debug.test_sampler_throughput |& tee tracy_sampler_nongreedy_tt_sampling.log
# TRACY_PROFILING_ACTIVE=1 python -m tracy -p -r -o tracy_sampler_nongreedy_tt_sampling_topk -m perf_debug.test_sampler_throughput -- --top-k 3 |& tee tracy_sampler_nongreedy_tt_sampling_topk.log



# OPriginal tests

# VLLM_ENABLE_V1_MULTIPROCESSING=0 python -m tracy -p -r --sync-host-device -o tracy_my_run -m pytest -svv --max-output-tokens 3 "tests/benchmark/test_vllm_benchmarks.py::test_vllm_trace_benchmark[opt-125m]" |& tee tracy_opt_125m.log

# python perf_debug/test_sampler_throughput.py

# TRACY_PROFILING_ACTIVE=1 python -m tracy -p -r -o tracy_sampler_nongreedy_baseline_topk -m perf_debug.test_sampler_throughput -- --top-k 3 |& tee tracy_sampler_nongreedy_baseline_topk.log

# Tracy
# TRACY_PROFILING_ACTIVE=1 python -m tracy -p -r --sync-host-device -o tracy_sampler_nongreedy_topk_chunking -m perf_debug.test_sampler_throughput |& tee tracy_sampler_nongreedy_topk_chunking.log
# TRACY_PROFILING_ACTIVE=1 python -m tracy -p -r --sync-host-device -o tracy_sampler_nongreedy_baseline -m perf_debug.test_sampler_throughput |& tee tracy_sampler_nongreedy_baseline.log


# Step 1

# TRACY_PROFILING_ACTIVE=1 python -m tracy -p -r -o tracy_sampler_nongreedy_topk_chunked_topk -m perf_debug.test_sampler_throughput -- --top-k 3 |& tee tracy_sampler_nongreedy_topk_chunked_topk.log
# TRACY_PROFILING_ACTIVE=1 python -m tracy -p -r -o tracy_sampler_nongreedy_topk_chunked -m perf_debug.test_sampler_throughput |& tee tracy_sampler_nongreedy_topk_chunked.log

# TRACY_PROFILING_ACTIVE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
#     python -m tracy -p -r \
#     -o tracy_e2e_llama1b_b2_non_topk_chunked \
#     -m pytest -svv \
#     --max-output-tokens 6 \
#     "tests/benchmark/test_vllm_benchmarks.py::test_sampling_quality[llama3.2-1b-b2-nongreedy-device]" \
#     |& tee tracy_e2e_llama1b_b2_non_topk_chunked.log


# Step 2

# TRACY_PROFILING_ACTIVE=1 python -m tracy -p -r -o tracy_sampler_nongreedy_tt_sampling -m perf_debug.test_sampler_throughput |& tee tracy_sampler_nongreedy_tt_sampling.log
# TRACY_PROFILING_ACTIVE=1 python -m tracy -p -r -o tracy_sampler_nongreedy_tt_sampling_topk -m perf_debug.test_sampler_throughput -- --top-k 3 |& tee tracy_sampler_nongreedy_tt_sampling_topk.log

# TRACY_PROFILING_ACTIVE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
#     python -m tracy -p -r \
#     -o tracy_e2e_llama1b_b2_tt_sampling \
#     -m pytest -svv \
#     --max-output-tokens 6 \
#     "tests/benchmark/test_vllm_benchmarks.py::test_sampling_quality[llama3.2-1b-b2-nongreedy-device]" \
#     |& tee tracy_e2e_llama1b_b2_tt_sampling.log