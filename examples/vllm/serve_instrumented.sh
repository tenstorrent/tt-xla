#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Launch any examples/vllm/<model>/service.sh with TT engine telemetry enabled,
# so the live dashboard can read TRUE per-slot state (Approach B):
#
#   examples/vllm/serve_instrumented.sh Llama-3.1-8B-Instruct [batch_size]
#
# Then, in another shell, point the dashboard at the telemetry dir it prints:
#
#   python3 integrations/vllm_plugin/tools/live_dashboard.py \
#       --source snapshot --dir <printed dir>
#
# Telemetry is env-gated (vllm_tt/instrumentation.py); with TT_INSTRUMENT=1 the
# engine writes snapshot.json + events.jsonl into TT_INSTRUMENT_DIR. The vLLM
# EngineCore runs as a subprocess, so the dir MUST be absolute to survive the
# subprocess's working directory -- this wrapper guarantees that.
#
# To play against the server WITHOUT touching it (Approach A, inferred state):
#   python3 integrations/vllm_plugin/tools/live_dashboard.py --source client \
#       --port 8000 --model meta-llama/Llama-3.1-8B-Instruct

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${1:?usage: serve_instrumented.sh <model-dir under examples/vllm> [batch_size]}"
BATCH="${2:-1}"
# Which launch script in the model dir to run (default service.sh). E.g.:
#   SERVICE_SCRIPT=service_chunked.sh examples/vllm/serve_instrumented.sh Qwen3-8B 32
SERVICE_SCRIPT="${SERVICE_SCRIPT:-service.sh}"

SERVICE="$REPO_ROOT/examples/vllm/$MODEL/$SERVICE_SCRIPT"
if [[ ! -f "$SERVICE" ]]; then
    echo "No $SERVICE_SCRIPT for '$MODEL' (looked at $SERVICE)" >&2
    echo "Available scripts:" >&2
    ls "$REPO_ROOT/examples/vllm/$MODEL"/*.sh 2>/dev/null >&2 || ls "$REPO_ROOT/examples/vllm" >&2
    exit 1
fi

# Absolute telemetry dir so the EngineCore subprocess writes where we expect.
export TT_INSTRUMENT=1
export TT_INSTRUMENT_DIR="${TT_INSTRUMENT_DIR:-/tmp/tt_instrument/$MODEL}"
# Throttle step snapshots to ~10/s; keep the hot path cheap (see #4278 lesson).
export TT_INSTRUMENT_THROTTLE_MS="${TT_INSTRUMENT_THROTTLE_MS:-100}"
mkdir -p "$TT_INSTRUMENT_DIR"

cat <<EOF
================================================================================
 Instrumented vLLM launch: $MODEL  (batch_size=$BATCH)
 Telemetry dir : $TT_INSTRUMENT_DIR   (snapshot.json + events.jsonl)
 Snapshot TUI  : python3 integrations/vllm_plugin/tools/live_dashboard.py \\
                     --source snapshot --dir $TT_INSTRUMENT_DIR
 Client TUI    : python3 integrations/vllm_plugin/tools/live_dashboard.py \\
                     --source client --port 8000 --model <served-model>
================================================================================
EOF

exec bash "$SERVICE" "$BATCH"
