#!/bin/bash
# Batch scaling sweep: greedy-device and non-greedy-device
# across OPT-125M, Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B
# at batch=1, 16, 32. Fix must be applied before running.
#
# Usage: bash perf_debug/run_batch_scaling_sweep.sh <output_dir>

set -euo pipefail

OUTDIR="${1:?Usage: $0 <output_dir>}"
mkdir -p "$OUTDIR"

# OPT-125M (vocab=50,272, gpu_mem=0.1 default)
python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 128 --temperature 0.0 --batch-size 1  --max-tokens 128 |& tee "$OUTDIR/opt125m_batch1_greedy_device.log"
python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 128 --temperature 0.8 --batch-size 1  --max-tokens 128 |& tee "$OUTDIR/opt125m_batch1_non_greedy_device.log"
python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 128 --temperature 0.0 --batch-size 16 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/opt125m_batch16_greedy_device.log"
python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 128 --temperature 0.8 --batch-size 16 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/opt125m_batch16_non_greedy_device.log"
python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 128 --temperature 0.0 --batch-size 32 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/opt125m_batch32_greedy_device.log"
python examples/vllm/opt-125m/chat.py --benchmark --max-model-len 128 --temperature 0.8 --batch-size 32 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/opt125m_batch32_non_greedy_device.log"

# Llama-3.2-1B
python examples/vllm/Llama-3.2-1B-Instruct/chat.py --benchmark --max-model-len 128 --temperature 0.0 --batch-size 1  --max-tokens 128 |& tee "$OUTDIR/llama3p2_1b_batch1_greedy_device.log"
python examples/vllm/Llama-3.2-1B-Instruct/chat.py --benchmark --max-model-len 128 --temperature 0.8 --batch-size 1  --max-tokens 128 |& tee "$OUTDIR/llama3p2_1b_batch1_non_greedy_device.log"
python examples/vllm/Llama-3.2-1B-Instruct/chat.py --benchmark --max-model-len 128 --temperature 0.0 --batch-size 16 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/llama3p2_1b_batch16_greedy_device.log"
python examples/vllm/Llama-3.2-1B-Instruct/chat.py --benchmark --max-model-len 128 --temperature 0.8 --batch-size 16 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/llama3p2_1b_batch16_non_greedy_device.log"
python examples/vllm/Llama-3.2-1B-Instruct/chat.py --benchmark --max-model-len 128 --temperature 0.0 --batch-size 32 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/llama3p2_1b_batch32_greedy_device.log"
python examples/vllm/Llama-3.2-1B-Instruct/chat.py --benchmark --max-model-len 128 --temperature 0.8 --batch-size 32 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/llama3p2_1b_batch32_non_greedy_device.log"

# Llama-3.2-3B
python examples/vllm/Llama-3.2-3B-Instruct/chat.py --benchmark --max-model-len 128 --temperature 0.0 --batch-size 1  --max-tokens 128 |& tee "$OUTDIR/llama3p2_3b_batch1_greedy_device.log"
python examples/vllm/Llama-3.2-3B-Instruct/chat.py --benchmark --max-model-len 128 --temperature 0.8 --batch-size 1  --max-tokens 128 |& tee "$OUTDIR/llama3p2_3b_batch1_non_greedy_device.log"
python examples/vllm/Llama-3.2-3B-Instruct/chat.py --benchmark --max-model-len 128 --temperature 0.0 --batch-size 16 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/llama3p2_3b_batch16_greedy_device.log"
python examples/vllm/Llama-3.2-3B-Instruct/chat.py --benchmark --max-model-len 128 --temperature 0.8 --batch-size 16 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/llama3p2_3b_batch16_non_greedy_device.log"
python examples/vllm/Llama-3.2-3B-Instruct/chat.py --benchmark --max-model-len 128 --temperature 0.0 --batch-size 32 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/llama3p2_3b_batch32_greedy_device.log"
python examples/vllm/Llama-3.2-3B-Instruct/chat.py --benchmark --max-model-len 128 --temperature 0.8 --batch-size 32 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/llama3p2_3b_batch32_non_greedy_device.log"

# Llama-3.1-8B
python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 128 --temperature 0.0 --batch-size 1  --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_batch1_greedy_device.log"
python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 128 --temperature 0.8 --batch-size 1  --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_batch1_non_greedy_device.log"
python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 128 --temperature 0.0 --batch-size 16 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_batch16_greedy_device.log"
python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 128 --temperature 0.8 --batch-size 16 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_batch16_non_greedy_device.log"
python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 128 --temperature 0.0 --batch-size 32 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_batch32_greedy_device.log"
python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --max-model-len 128 --temperature 0.8 --batch-size 32 --gpu-memory-utilization 0.037 --max-tokens 128 |& tee "$OUTDIR/llama3p1_8b_batch32_non_greedy_device.log"
