#!/usr/bin/env bash
# Perf sweep: Llama 1B / 8B at batch ∈ {1, 2, 8, 16, 32}, three variants each.
#
# Usage:
#   bash run_perf_sweep.sh                                 # default settings
#   TT_USE_TTNN_SAMPLING=0 bash run_perf_sweep.sh          # disable ttnn.sampling
#
# Goal: measure where sampler stops being on the critical path as batch grows,
# and quantify the e2e impact of the all_random skip + future batch-padding
# elimination work. Subset of test_sampling_quality scope (only Llama 1B/8B,
# 4 batch sizes, 3 variants = 24 configs). Drops greedy-cpu (only differs from
# greedy-device by where argmax runs; not relevant for sampler perf decisions).

# No set -e — we want all configs to run even if a few fail.

LOG_DIR="perf_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
echo "Logs will go to: $LOG_DIR"

run() {
    local id="$1"
    echo ""
    echo "===== $id ====="
    pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_vllm_perf_sweep[$id]" \
        |& tee "$LOG_DIR/${id}.log"
}

for bs in 1 2 8 16 32; do
    for model in llama3.2-1b llama3.1-8b; do
        for variant in greedy-device nongreedy-device nongreedy-cpu; do
            run "${model}-b${bs}-${variant}"
        done
    done
done

echo ""
echo "===== tok/s Summary ====="
printf "%-50s %s\n" "config" "tok/s"
printf "%-50s %s\n" "------" "-----"
for log in "$LOG_DIR"/*.log; do
    name=$(basename "$log" .log)
    tps=$(grep "Sample per second" "$log" | awk '{print $NF}' | head -1)
    printf "%-50s %s\n" "$name" "${tps:-FAILED}"
done | sort

echo ""
echo "Logs in: $LOG_DIR"
