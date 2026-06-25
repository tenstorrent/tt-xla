# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Qwen3-8B served with the chunked-prefill / b1-prefill "best settings" from the
# benchmark repro (tests/benchmark::test_vllm_qwen3_8b_kv_slowdown, commit
# 6fd7d73ca): batch-32 server graph, chunked prefill (2048), the b1-prefill graph
# routed by a prefill-batch threshold of 16, 40K max context, optimization_level
# 1, trace on, BFP8 weights + KV cache.
#
# Knobs (env vars):
#   BATCH_SIZE     max_num_seqs / batch slots         (default 32)
#   MAX_MODEL_LEN  max context length                 (default 40960)
#   NUM_LAYERS     num_hidden_layers override; 0=full (default 0)
#                  -> set NUM_LAYERS=1 for a fast compile while bringing things up
#   PORT           server port                        (default 8000)
#
# Quick bring-up (1 layer, compiles fast), then watch it live:
#   NUM_LAYERS=1 TT_INSTRUMENT=1 TT_INSTRUMENT_DIR=/tmp/tt_instrument/qwen3-8b \
#       bash examples/vllm/Qwen3-8B/service_chunked.sh
#   # another shell:
#   python3 integrations/vllm_plugin/tools/live_dashboard.py \
#       --source snapshot --dir /tmp/tt_instrument/qwen3-8b
#
# Or launch instrumented via the wrapper (sets TT_INSTRUMENT + an absolute dir):
#   SERVICE_SCRIPT=service_chunked.sh examples/vllm/serve_instrumented.sh Qwen3-8B 32
#
# Talk to it: examples/vllm/Qwen3-8B/client.py (interactive, --num-users N), or
# the dashboard's client source for inferred visualization with no telemetry:
#   python3 integrations/vllm_plugin/tools/live_dashboard.py --source client \
#       --port 8000 --model Qwen/Qwen3-8B

set -u

BATCH_SIZE="${BATCH_SIZE:-${1:-32}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
NUM_LAYERS="${NUM_LAYERS:-0}"
PORT="${PORT:-8000}"

# additional-config: real JSON types (TTConfig is a plain dataclass -- a quoted
# "false" would be a truthy string). Mirrors the benchmark's server_like config.
ADD='{'
ADD+='"optimization_level": 1, "enable_trace": true, "enable_const_eval": true, '
ADD+='"min_context_len": 128, "prefill_chunk_size": 2048, '
ADD+='"prefill_batch_threshold": 16, "min_num_seqs": 1, '
ADD+='"experimental_weight_dtype": "bfp_bf8", '
ADD+='"experimental_kv_cache_dtype": "bfp_bf8", "fp32_dest_acc_en": false'
if [ "$NUM_LAYERS" -gt 0 ]; then
    ADD+=", \"num_hidden_layers\": $NUM_LAYERS"
fi
ADD+='}'

echo "Qwen3-8B (chunked prefill + b1-prefill): batch=$BATCH_SIZE max_model_len=$MAX_MODEL_LEN num_layers=${NUM_LAYERS:-full} port=$PORT"
echo "additional-config: $ADD"
[ "${TT_INSTRUMENT:-0}" = "1" ] && echo "telemetry ON -> ${TT_INSTRUMENT_DIR:-<cwd>/.tt_instrument}"

vllm serve Qwen/Qwen3-8B \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$BATCH_SIZE" \
    --max-num-batched-tokens 2048 \
    --no-enable-prefix-caching \
    --gpu-memory-utilization 0.30 \
    --port "$PORT" \
    --override-generation-config '{"chat_template_kwargs": {"enable_thinking": false}}' \
    --additional-config "$ADD"
