#!/usr/bin/env bash
# Qwen3-8B KV-depth decode/prefill slowdown repro.
#
# Runs 6 datapoints (3 shapes x batch-32 and batch-1) of the single env-driven
# test tests/benchmark/test_vllm_benchmarks.py::test_vllm_qwen3_8b_kv_slowdown.
# Each pytest command below is self-contained (shape set inline via env vars) so
# the lines can be copy-pasted individually as reference commands in an issue.
#
# Three shapes, all reaching ~16K KV-cache depth two different ways:
#   ISL128  + OSL128    baseline (shallow)
#   ISL16K  + OSL128    KV built from PREFILL  (prefill-heavy; the case to debug)
#   ISL128  + OSL16K    KV built from DECODE   (decode walks KV up to ~16K)
# Prefix caching is OFF and warmup is capped at 128 tokens (handled by the test).
#
# Run from the tt-xla venv, with the device + mesh descriptor set in the env:
#   cd /home/kmabee/tt-xla && source venv/activate
#   export TT_VISIBLE_DEVICES=0
#   export TT_MESH_GRAPH_DESC_PATH=<.../p150_mesh_graph_descriptor.textproto>
#   ./run_kv_slowdown.sh
set -u

# NOTE: On a multichip machine (e.g. QB2) set TT_VISIBLE_DEVICES and
# TT_MESH_GRAPH_DESC_PATH in the calling environment to pin one device and point
# at its mesh descriptor. On a single-chip machine neither is needed.

# One timestamped dir for all logs from this invocation.
DIR="perf_sweeps/kv_slowdown_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DIR"
echo "Logs -> $DIR"

# ===================== IR snapshot (TTXLA_LOGGER_LEVEL=DEBUG, ISL128/OSL128) =====================
# One DEBUG run per batch at the smallest shape to dump the IR graphs to the log
# (also serialized to modules/). The IR is identical across shapes at this
# max_model_len, so the remaining (perf) runs below skip DEBUG to keep logs small.
TTXLA_LOGGER_LEVEL=DEBUG TT_BENCHMARK_BATCH_SIZE=32 TT_BENCHMARK_MAX_MODEL_LEN=40960 TT_BENCHMARK_ISL=128 TT_BENCHMARK_OSL=128 pytest -svv tests/benchmark/test_vllm_benchmarks.py -k test_vllm_qwen3_8b_kv_slowdown 2>&1 | tee "$DIR/b32_isl128_osl128_DEBUG.log"
TTXLA_LOGGER_LEVEL=DEBUG TT_BENCHMARK_BATCH_SIZE=1  TT_BENCHMARK_MAX_MODEL_LEN=40960 TT_BENCHMARK_ISL=128 TT_BENCHMARK_OSL=128 pytest -svv tests/benchmark/test_vllm_benchmarks.py -k test_vllm_qwen3_8b_kv_slowdown 2>&1 | tee "$DIR/b1_isl128_osl128_DEBUG.log"

# ===================== batch-32 (server graph, max_model_len=40960) =====================
TT_BENCHMARK_BATCH_SIZE=32 TT_BENCHMARK_MAX_MODEL_LEN=40960 TT_BENCHMARK_ISL=128   TT_BENCHMARK_OSL=128   pytest -svv tests/benchmark/test_vllm_benchmarks.py -k test_vllm_qwen3_8b_kv_slowdown 2>&1 | tee "$DIR/b32_isl128_osl128.log"
TT_BENCHMARK_BATCH_SIZE=32 TT_BENCHMARK_MAX_MODEL_LEN=40960 TT_BENCHMARK_ISL=16384 TT_BENCHMARK_OSL=128   pytest -svv tests/benchmark/test_vllm_benchmarks.py -k test_vllm_qwen3_8b_kv_slowdown 2>&1 | tee "$DIR/b32_isl16384_osl128.log"
TT_BENCHMARK_BATCH_SIZE=32 TT_BENCHMARK_MAX_MODEL_LEN=40960 TT_BENCHMARK_ISL=128   TT_BENCHMARK_OSL=16384 pytest -svv tests/benchmark/test_vllm_benchmarks.py -k test_vllm_qwen3_8b_kv_slowdown 2>&1 | tee "$DIR/b32_isl128_osl16384.log"

# ===================== batch-1 (b1-prefill graph, max_model_len=40960 to match b32) =====================
TT_BENCHMARK_BATCH_SIZE=1 TT_BENCHMARK_MAX_MODEL_LEN=40960 TT_BENCHMARK_ISL=128   TT_BENCHMARK_OSL=128   pytest -svv tests/benchmark/test_vllm_benchmarks.py -k test_vllm_qwen3_8b_kv_slowdown 2>&1 | tee "$DIR/b1_isl128_osl128.log"
TT_BENCHMARK_BATCH_SIZE=1 TT_BENCHMARK_MAX_MODEL_LEN=40960 TT_BENCHMARK_ISL=16384 TT_BENCHMARK_OSL=128   pytest -svv tests/benchmark/test_vllm_benchmarks.py -k test_vllm_qwen3_8b_kv_slowdown 2>&1 | tee "$DIR/b1_isl16384_osl128.log"
TT_BENCHMARK_BATCH_SIZE=1 TT_BENCHMARK_MAX_MODEL_LEN=40960 TT_BENCHMARK_ISL=128   TT_BENCHMARK_OSL=16384 pytest -svv tests/benchmark/test_vllm_benchmarks.py -k test_vllm_qwen3_8b_kv_slowdown 2>&1 | tee "$DIR/b1_isl128_osl16384.log"

echo "Done. Logs in $DIR"
