#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Regenerate single-layer LLM benchmark tests (TTIR/TTNN MLIRs + per-test JSON).
#
# Env vars:
#   TT_MLIR_COMMIT_OVERRIDE  Pin tt-mlir to this SHA and rebuild first.
#                            Output dirs get a SHA suffix to avoid collisions.
#   TT_MLIR_LOCAL_PATH       Required when the override SHA only exists
#                            locally (rebased commit). Forwarded to rebuilder.
#   SUBSET                   Comma-list of {single,llmbox,galaxy} from
#                            tests/benchmark/single_layer/subsets.sh.
#                            Default: single.
#   HF_TOKEN                 Required; env or interactive prompt.
#
# Recovery loop. Three retryable runner exits:
#   42       runner self-reported "device needs reset"; we run safe_reset.
#   137/143  orchestrator was killed externally (e.g. user ran safe_reset
#            interactively and chose to kill the python ancestor too). The
#            device was already reset in that path; just relaunch with --continue.
# Other non-zero exits stop the sweep.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTXLA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SCRIPTS_DIR="$TTXLA_ROOT/scripts"
RUNNER="$SCRIPT_DIR/run_benchmarks.py"
SUBSETS_FILE="$SCRIPT_DIR/subsets.sh"

OVERRIDE="${TT_MLIR_COMMIT_OVERRIDE:-}"
LOCAL_PATH="${TT_MLIR_LOCAL_PATH:-}"
SUBSET="${SUBSET:-single}"

[[ -n "$LOCAL_PATH" && -z "$OVERRIDE" ]] && {
    echo "ERROR: TT_MLIR_LOCAL_PATH set without TT_MLIR_COMMIT_OVERRIDE." >&2; exit 2; }

# Default cache locations are under $HOME, which has limited space; redirect
# to tt-xla's dir (typically /localdev).
export HF_HOME="${HF_HOME:-$TTXLA_ROOT/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$TTXLA_ROOT/.cache/pip}"
export TT_METAL_CACHE="${TT_METAL_CACHE:-$TTXLA_ROOT/.cache/tt-metal-cache}"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TT_METAL_CACHE"

source "$SUBSETS_FILE"
TESTS=""; sl=0; sg=0
for n in ${SUBSET//,/ }; do
    case "$n" in
        single) TESTS+="${TESTS:+,}$SUBSET_SINGLE" ;;
        llmbox) TESTS+="${TESTS:+,}$SUBSET_LLMBOX"; sl=1 ;;
        galaxy) TESTS+="${TESTS:+,}$SUBSET_GALAXY"; sg=1 ;;
        *) echo "ERROR: unknown SUBSET '$n' (single,llmbox,galaxy)." >&2; exit 2 ;;
    esac
done
(( sl && sg )) && { echo "ERROR: llmbox + galaxy can't combine (different meshes)." >&2; exit 2; }

if [[ -z "${HF_TOKEN:-}" ]]; then
    [[ -t 0 ]] || { echo "ERROR: HF_TOKEN not set, stdin not a TTY." >&2; exit 2; }
    read -rsp "HF_TOKEN (input hidden): " HF_TOKEN; echo
    export HF_TOKEN
fi

SUFFIX="${OVERRIDE:+_${OVERRIDE:0:8}}"
TTIRS_DIR="${SUFFIX:+$SCRIPT_DIR/generated${SUFFIX}/ttir}"
TTNN_DIR="${SUFFIX:+$SCRIPT_DIR/generated${SUFFIX}/ttnn}"

cd "$TTXLA_ROOT"
set +u; source venv/activate; set -u

[[ -n "$OVERRIDE" ]] && "$SCRIPTS_DIR/rebuild_for_custom_mlir.sh" "$OVERRIDE" "$LOCAL_PATH"

ARGS=()
[[ -n "$TESTS"     ]] && ARGS+=(--test "$TESTS")
[[ -n "$TTIRS_DIR" ]] && ARGS+=(--ttirs-output-dir "$TTIRS_DIR")
[[ -n "$TTNN_DIR"  ]] && ARGS+=(--ttnn-output-dir  "$TTNN_DIR")

attempt=0
while :; do
    set +e; python "$RUNNER" "${ARGS[@]}"; rc=$?; set -e
    [[ $rc -eq 0 ]] && break
    case $rc in
        42)        echo "[regen] device needs reset ($((attempt+1))/3); resetting..."
                   "$SCRIPTS_DIR/safe_reset.sh" --force ;;
        137|143)   echo "[regen] orchestrator killed externally (rc=$rc); assuming safe_reset already ran." ;;
        *)         echo "[regen] runner exited $rc; stopping."; break ;;
    esac
    (( ++attempt > 3 )) && { echo "[regen] gave up after 3 resets."; break; }
    [[ "${ARGS[*]}" != *--continue* ]] && ARGS+=(--continue)
done
