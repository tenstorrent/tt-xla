#!/bin/bash
DIR=apr12_sampling_comparison_improvements
mkdir -p $DIR
# pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-greedy-device]" |& tee $DIR/8b-b1-greedy-device.log
# pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-greedy-cpu]" |& tee $DIR/8b-b1-greedy-cpu.log
# pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-nongreedy-device]" |& tee $DIR/8b-b1-nongreedy-device.log
# pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-nongreedy-cpu]" |& tee $DIR/8b-b1-nongreedy-cpu.log
# pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b32-greedy-device]" |& tee $DIR/8b-b32-greedy-device.log
# pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b32-greedy-cpu]" |& tee $DIR/8b-b32-greedy-cpu.log
# pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b32-nongreedy-device]" |& tee $DIR/8b-b32-nongreedy-device.log
# pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b32-nongreedy-cpu]" |& tee $DIR/8b-b32-nongreedy-cpu.log

# pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-nongreedy-device]" |& tee $DIR/8b-b1-nongreedy-device-scatter_fix.log
# pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b32-nongreedy-device]" |& tee $DIR/8b-b32-nongreedy-device-scatter_fix.log

# pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-nongreedy-device]" |& tee $DIR/8b-b1-nongreedy-device-v2_trace.log
# pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b32-nongreedy-device]" |& tee $DIR/8b-b32-nongreedy-device-v2_trace.log

# ttnn.sampling integration: non-greedy with fused sampling kernel
# TT_USE_TTNN_SAMPLING=1 pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-nongreedy-device]" |& tee $DIR/8b-b1-nongreedy-device-ttnn_sampling.log
# TT_USE_TTNN_SAMPLING=1 pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b32-nongreedy-device]" |& tee $DIR/8b-b32-nongreedy-device-ttnn_sampling.log

# ttnn.sampling with trace enabled
TT_USE_TTNN_SAMPLING=1 pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison_trace[8b-b1-nongreedy-device-trace]" |& tee $DIR/8b-b1-nongreedy-device-ttnn_sampling_trace.log
TT_USE_TTNN_SAMPLING=1 pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison_trace[8b-b32-nongreedy-device-trace]" |& tee $DIR/8b-b32-nongreedy-device-ttnn_sampling_trace.log
