#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

MODELS=(
    "test_llama_3_1_8b:llama_3_1_8b"
    # "test_qwen_3_8b:qwen_3_8b"
    # "test_llama_3_2_1b:llama_3_2_1b"
    # "test_gemma_1_1_2b:gemma_1_1_2b"
    # "test_phi1:phi1"
    #"test_llama_3_1_70b_tp:llama_3_1_70b"
)

RESULTS_DIR="results_$(date +%Y%m%d_%H%M%S)"
REPORT="${RESULTS_DIR}/perf_report.md"
mkdir -p "$RESULTS_DIR"

printf "| model | bs | seq | opt | ttft_ms | device_ms | host_ms |\n" > "$REPORT"
printf "|-------|----|-----|-----|---------|-----------|----------|\n" >> "$REPORT"

# Parses ops_perf_results.csv for a single run dir.
# Outputs: ttft_ms device_ms host_ms
parse_metrics() {
    local ops_csv="$1"
    python3 - "$ops_csv" <<'EOF'
import sys, csv

ops_csv = sys.argv[1]

signposts = []
ops = []
with open(ops_csv) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if not row or not row[10]:
            continue
        try:
            ts = int(row[10])
        except ValueError:
            continue
        if row[1] == 'signpost':
            signposts.append((row[0], ts))
        else:
            try:
                dev_kern = int(row[18]) if row[18] else 0
            except ValueError:
                dev_kern = 0
            ops.append((ts, dev_kern))

# Measurement window: first prefill_start/end pair after `warmup_complete`.
# Skips the warmup iterations so we never report stale warmup numbers when
# the real measurement run crashes.
warmup_idx = next(
    (i for i, (n, _) in enumerate(signposts) if 'warmup_complete' in n), None
)
if warmup_idx is None:
    print("0 0 0")
    sys.exit(0)

s = next(
    (ts for n, ts in signposts[warmup_idx + 1:] if 'prefill_start' in n), None
)
if s is None:
    print("0 0 0")
    sys.exit(0)

e = next(
    (ts for n, ts in signposts[warmup_idx + 1:]
     if 'prefill_end' in n and ts > s),
    None,
)
if e is None:
    op_ts_after = [t for t, _ in ops if t >= s]
    if not op_ts_after:
        print("0 0 0")
        sys.exit(0)
    e = max(op_ts_after)

ttft_ns = e - s
device_ns = sum(d for t, d in ops if s <= t <= e)
host_ns = max(ttft_ns - device_ns, 0)

print(f"{ttft_ns/1e6:.1f} {device_ns/1e6:.1f} {host_ns/1e6:.1f}")
EOF
}

for model_entry in "${MODELS[@]}"; do
    test_fn="${model_entry%%:*}"
    model_name="${model_entry##*:}"
    for opt in 0 1; do
        for bs in 1 32; do
            for seq in 1024 128; do
                tt-smi -r || true
                out_dir="${RESULTS_DIR}/${model_name}_bs${bs}_seq${seq}_opt${opt}"
                mkdir -p "$out_dir"

                # NOTE: do NOT pass --sync-host-device here. It sets
                # TT_METAL_PROFILER_SYNC=1, which makes tt-metal run ProfilerSync(INIT)
                # while opening the mesh device. That launches a program before the
                # device is fully initialized (mesh_device.cpp:422 runs before
                # initialize_fabric_and_dispatch_fw at :423), tripping
                # TT_FATAL(device->is_initialized()) in program.cpp. Device-side op
                # profiling (ops_perf_results*.csv) still works without it.
                TT_METAL_DEVICE_PROFILER=1 python3 -m tracy -v -r -p \
                    -o "${out_dir}/tracy" \
                    --tracy-tools-folder "$(pwd)/third_party/tt-mlir/install/bin" \
                    -m "pytest tests/benchmark/test_llms.py::${test_fn} \
                        --prefill-only \
                        --batch-size $bs \
                        --input-sequence-length $seq \
                        --output-file ${out_dir}/metrics.json \
                        --optimization-level ${opt} -svv"

                ops_csv=$(find "${out_dir}/tracy/reports" -name "ops_perf_results*.csv" | head -1)

                read ttft_ms device_ms host_ms < <(parse_metrics "$ops_csv")
                printf "| %s | %s | %s | %s | %s | %s | %s |\n" \
                    "$model_name" "$bs" "$seq" "$opt" \
                    "$ttft_ms" "$device_ms" "$host_ms" >> "$REPORT"
            done
        done
    done
done

echo "Report written to $REPORT"
