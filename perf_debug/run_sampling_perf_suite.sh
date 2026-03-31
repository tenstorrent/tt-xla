#!/bin/bash
# Sampling performance benchmark suite.
# Runs 12 configs: OPT-125M and Llama-3.1-8B at seq_len=128/2048,
# greedy device / non-greedy CPU / non-greedy device.
#
# Usage: bash perf_debug/run_sampling_perf_suite.sh <output_dir>
# Example:
#   bash perf_debug/run_sampling_perf_suite.sh perf_results/before_fix
#   bash perf_debug/run_sampling_perf_suite.sh perf_results/after_fix

set -euo pipefail

OUTDIR="${1:?Usage: $0 <output_dir>}"
mkdir -p "$OUTDIR"

TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 128  --temperature 0.0 --max-tokens 128 |& tee "$OUTDIR/opt125m_seq128_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 128  --temperature 0.8 --cpu-sampling --max-tokens 128 |& tee "$OUTDIR/opt125m_seq128_non_greedy_cpu.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 128  --temperature 0.8 --max-tokens 128 |& tee "$OUTDIR/opt125m_seq128_non_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 2048 --temperature 0.0 --max-tokens 128 |& tee "$OUTDIR/opt125m_seq2048_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 2048 --temperature 0.8 --cpu-sampling --max-tokens 128 |& tee "$OUTDIR/opt125m_seq2048_non_greedy_cpu.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 2048 --temperature 0.8 --max-tokens 128 |& tee "$OUTDIR/opt125m_seq2048_non_greedy_device.log"

TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 128  --temperature 0.0 --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_seq128_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 128  --temperature 0.8 --cpu-sampling --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_seq128_non_greedy_cpu.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 128  --temperature 0.8 --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_seq128_non_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 2048 --temperature 0.0 --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_seq2048_greedy_device.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 2048 --temperature 0.8 --cpu-sampling --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_seq2048_non_greedy_cpu.log"
TTXLA_LOGGER_LEVEL=DEBUG python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 2048 --temperature 0.8 --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_seq2048_non_greedy_device.log"
