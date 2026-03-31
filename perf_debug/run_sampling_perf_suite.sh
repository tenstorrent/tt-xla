#!/bin/bash
# Sampling performance benchmark suite.
# Runs greedy device / non-greedy CPU / non-greedy device for:
#   - OPT-125M, Llama-3.2-3B, Llama-3.1-8B
#   - seq_len=128 and seq_len=2048
#   - batch=1 (all configs) + batch=16 and batch=32 (non-greedy device only)
#
# Tracking issue: https://github.com/tenstorrent/tt-xla/issues/3940
#
# Usage: bash perf_debug/run_sampling_perf_suite.sh <output_dir>
# Example:
#   bash perf_debug/run_sampling_perf_suite.sh perf_results/before_fix
#   bash perf_debug/run_sampling_perf_suite.sh perf_results/after_fix
#
# The caller is responsible for applying or reverting the fix before running.
# gpu_memory_utilization=0.037 is used for batch>=16 (matches test_vllm_benchmarks.py).

set -euo pipefail

OUTDIR="${1:?Usage: $0 <output_dir>}"
mkdir -p "$OUTDIR"

# OPT-125M — batch=1
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 128  --temperature 0.0 --max-tokens 128 |& tee "$OUTDIR/opt125m_seq128_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 128  --temperature 0.8 --cpu-sampling --max-tokens 128 |& tee "$OUTDIR/opt125m_seq128_non_greedy_cpu.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 128  --temperature 0.8 --max-tokens 128 |& tee "$OUTDIR/opt125m_seq128_non_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 2048 --temperature 0.0 --max-tokens 128 |& tee "$OUTDIR/opt125m_seq2048_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 2048 --temperature 0.8 --cpu-sampling --max-tokens 128 |& tee "$OUTDIR/opt125m_seq2048_non_greedy_cpu.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 2048 --temperature 0.8 --max-tokens 128 |& tee "$OUTDIR/opt125m_seq2048_non_greedy_device.log"

# OPT-125M — batch=16 and batch=32 (non-greedy device only)
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 128  --temperature 0.8 --batch-size 16 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/opt125m_seq128_batch16_non_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 128  --temperature 0.8 --batch-size 32 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/opt125m_seq128_batch32_non_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 2048 --temperature 0.8 --batch-size 16 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/opt125m_seq2048_batch16_non_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 2048 --temperature 0.8 --batch-size 32 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/opt125m_seq2048_batch32_non_greedy_device.log"

# Llama-3.2-3B — batch=1
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.2-3B-Instruct/chat.py --benchmark --max-model-len 128  --temperature 0.0 --max-tokens 128 |& tee "$OUTDIR/llama3p2_3b_seq128_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.2-3B-Instruct/chat.py --benchmark --max-model-len 128  --temperature 0.8 --cpu-sampling --max-tokens 128 |& tee "$OUTDIR/llama3p2_3b_seq128_non_greedy_cpu.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.2-3B-Instruct/chat.py --benchmark --max-model-len 128  --temperature 0.8 --max-tokens 128 |& tee "$OUTDIR/llama3p2_3b_seq128_non_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.2-3B-Instruct/chat.py --benchmark --max-model-len 2048 --temperature 0.0 --max-tokens 128 |& tee "$OUTDIR/llama3p2_3b_seq2048_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.2-3B-Instruct/chat.py --benchmark --max-model-len 2048 --temperature 0.8 --cpu-sampling --max-tokens 128 |& tee "$OUTDIR/llama3p2_3b_seq2048_non_greedy_cpu.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.2-3B-Instruct/chat.py --benchmark --max-model-len 2048 --temperature 0.8 --max-tokens 128 |& tee "$OUTDIR/llama3p2_3b_seq2048_non_greedy_device.log"

# Llama-3.2-3B — batch=16 and batch=32 (non-greedy device only)
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.2-3B-Instruct/chat.py --benchmark --max-model-len 128  --temperature 0.8 --batch-size 16 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/llama3p2_3b_seq128_batch16_non_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.2-3B-Instruct/chat.py --benchmark --max-model-len 128  --temperature 0.8 --batch-size 32 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/llama3p2_3b_seq128_batch32_non_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.2-3B-Instruct/chat.py --benchmark --max-model-len 2048 --temperature 0.8 --batch-size 16 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/llama3p2_3b_seq2048_batch16_non_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.2-3B-Instruct/chat.py --benchmark --max-model-len 2048 --temperature 0.8 --batch-size 32 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/llama3p2_3b_seq2048_batch32_non_greedy_device.log"

# Llama-3.1-8B — batch=1
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 128  --temperature 0.0 --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_seq128_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 128  --temperature 0.8 --cpu-sampling --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_seq128_non_greedy_cpu.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 128  --temperature 0.8 --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_seq128_non_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 2048 --temperature 0.0 --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_seq2048_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 2048 --temperature 0.8 --cpu-sampling --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_seq2048_non_greedy_cpu.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 2048 --temperature 0.8 --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_seq2048_non_greedy_device.log"

# Llama-3.1-8B — batch=16 and batch=32 (non-greedy device only)
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 128  --temperature 0.8 --batch-size 16 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_seq128_batch16_non_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 128  --temperature 0.8 --batch-size 32 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_seq128_batch32_non_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 2048 --temperature 0.8 --batch-size 16 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_seq2048_batch16_non_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 2048 --temperature 0.8 --batch-size 32 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_seq2048_batch32_non_greedy_device.log"
