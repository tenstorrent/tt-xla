TTMLIR_ENABLE_PERF_TRACE=1 TTXLA_LOGGER_LEVEL=DEBUG TT_BENCHMARK_MAX_MODEL_LEN=128 TT_BENCHMARK_MAX_TOKENS=32 \
    pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_vllm_benchmark[llama-3.2-3b-batch32]" \
    |& tee perf_bench/llama-3.2-3b-batch32_seq128.log

TTMLIR_ENABLE_PERF_TRACE=1 TTXLA_LOGGER_LEVEL=DEBUG TT_BENCHMARK_MAX_MODEL_LEN=1024 TT_BENCHMARK_MAX_TOKENS=32 \
    pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_vllm_benchmark[llama-3.2-3b-batch32]" \
    |& tee perf_bench/llama-3.2-3b-batch32_seq1024.log

VLLM_ENABLE_V1_MULTIPROCESSING=0 TTXLA_LOGGER_LEVEL=DEBUG TT_BENCHMARK_MAX_MODEL_LEN=128 TT_BENCHMARK_MAX_TOKENS=16 \
    python -m tracy -p -r --sync-host-device \
    -o tracy_output_128 -m pytest -svv \
    "tests/benchmark/test_vllm_benchmarks.py::test_vllm_benchmark[llama-3.2-3b]" |& tee perf_bench/llama-3.2-3b_seq128.log

VLLM_ENABLE_V1_MULTIPROCESSING=0 TTXLA_LOGGER_LEVEL=DEBUG TT_BENCHMARK_MAX_MODEL_LEN=1024 TT_BENCHMARK_MAX_TOKENS=16 \
    python -m tracy -p -r --sync-host-device \
    -o tracy_output_1024 -m pytest -svv \
    "tests/benchmark/test_vllm_benchmarks.py::test_vllm_benchmark[llama-3.2-3b]" |& tee perf_bench/llama-3.2-3b_seq1024.log


# Re-capture (same commands as before)
rm -rf tracy_output_128 && VLLM_ENABLE_V1_MULTIPROCESSING=0 TT_BENCHMARK_MAX_MODEL_LEN=128  TT_BENCHMARK_MAX_TOKENS=16 \
    python -m tracy -p -r --sync-host-device \
    -o tracy_output_128 -m pytest -svv \
    "tests/benchmark/test_vllm_benchmarks.py::test_vllm_benchmark[llama-3.2-3b]" \
    |& tee perf_bench/llama-3.2-3b-seq128.log

rm -rf tracy_output_1024 && VLLM_ENABLE_V1_MULTIPROCESSING=0 TT_BENCHMARK_MAX_MODEL_LEN=1024 TT_BENCHMARK_MAX_TOKENS=16 \
    python -m tracy -p -r --sync-host-device \
    -o tracy_output_1024 -m pytest -svv \
    "tests/benchmark/test_vllm_benchmarks.py::test_vllm_benchmark[llama-3.2-3b-batch32]" \
    |& tee perf_bench/llama-3.2-3b-batch32_seq1024.log

# Filter to decode-only window for each
python /tmp/decode_filter.py perf_bench/llama-3.2-3b_batch32_seq128.log \
    tracy_output_128/reports/*/ops_perf_results_*.csv

python /tmp/decode_filter.py perf_bench/llama-3.2-3b_batch32_seq1024.log \
    tracy_output_1024/reports/*/ops_perf_results_*.csv


TT_BENCHMARK_MAX_MODEL_LEN=128 \
    pytest -svv \
    "tests/benchmark/test_vllm_benchmarks.py::test_vllm_benchmark[llama-3.2-3b-batch32]" \
    |& tee perf_bench/llama-3.2-3b-batch32_seq128.log

TT_BENCHMARK_MAX_MODEL_LEN=1024 \
    pytest -svv \
    "tests/benchmark/test_vllm_benchmarks.py::test_vllm_benchmark[llama-3.2-3b-batch32]" \
    |& tee perf_bench/llama-3.2-3b-batch32_seq1024.log