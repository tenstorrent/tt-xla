#!/bin/bash
DIR=apr12_sampling_comparison_improvements
mkdir -p $DIR
#pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-greedy-device]" |& tee $DIR/8b-b1-greedy-device.log
#pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-greedy-cpu]" |& tee $DIR/8b-b1-greedy-cpu.log
#pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-nongreedy-device]" |& tee $DIR/8b-b1-nongreedy-device.log
#pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-nongreedy-cpu]" |& tee $DIR/8b-b1-nongreedy-cpu.log
#pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b32-greedy-device]" |& tee $DIR/8b-b32-greedy-device.log
#pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b32-greedy-cpu]" |& tee $DIR/8b-b32-greedy-cpu.log
#pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b32-nongreedy-device]" |& tee $DIR/8b-b32-nongreedy-device.log
#pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b32-nongreedy-cpu]" |& tee $DIR/8b-b32-nongreedy-cpu.log

pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b1-nongreedy-device]" |& tee $DIR/8b-b1-nongreedy-device-initial_improve.log
pytest -svv --durations=0 "tests/benchmark/test_vllm_benchmarks.py::test_sampling_comparison[8b-b32-nongreedy-device]" |& tee $DIR/8b-b32-nongreedy-device-initial_improve.log


