#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Loop wrapper around ./run_falcon3_7b_evals.sh for hang-hunting: repeat the
# same eval invocation N times back-to-back, keeping the master log to one
# START and one result line per iteration while each iteration's full lm_eval
# output (including its per-task --limit/--concurrent, etc.) goes to
# <dir>/iter_NN.log. Mirrors the --loops/--dir/--timeout pattern added to
# ~/scripts/model_servers/run_evals_forge.sh (feea715, 2026-07-26).
#
# Usage:
#   ./run_falcon3_7b_evals_loop.sh --loops 20 --dir falcon3_loop |& tee master.log
#   ./run_falcon3_7b_evals_loop.sh --loops 10 --dir falcon3_loop --limit 1.0
#   ./run_falcon3_7b_evals_loop.sh --loops 5  --dir falcon3_loop --tasks ifeval --concurrent 8
#
# Verified single-layer hang-hunt run (2026-07-27, no hang across 10 loops):
#   ./run_falcon3_7b_evals_loop.sh --loops 10 --limit 0.75 \
#     --dir falcon3_loop_single_layer_0.75 \
#     |& tee falcon3_loop_master_single_layer_0.75.log
#
# Loop options (--flag value or --flag=value):
#   --loops N      run the whole eval invocation N times back-to-back (default 1).
#                  A hang shows up as an iteration that never prints a result
#                  line -- that is the one to attach gdb/py-spy to.
#   --dir DIR      mkdir -p DIR and send each iteration's full output to
#                  DIR/iter_NN.log, plus its lm_eval --output_path to
#                  DIR/iter_NN_results/ (unless --output is given explicitly,
#                  in which case every iteration reuses that one dir as-is).
#                  Relative paths resolve against your invoking cwd. Without
#                  --dir, iteration output is interleaved into the master log.
#   --timeout SECS abort an iteration that exceeds SECS and stop the loop
#                  (no default -- unset means wait forever).
#                  WARNING for #4521-class hangs: this SIGTERMs
#                  run_falcon3_7b_evals.sh, which disconnects the eval client.
#                  A client disconnect aborts in-flight requests and can
#                  force-clear the server's wedged state -- i.e. it can destroy
#                  the evidence you are trying to catch. To catch a hang in
#                  progress, run WITHOUT --timeout and let the loop block on
#                  the wedged iteration.
#
# All other flags (--tasks --limit --ifeval-limit --gpqa-limit --port
# --server-url --concurrent --model --output) pass straight through to
# run_falcon3_7b_evals.sh -- see that script's --help for their meaning.
#
# -h, --help     show this help
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$ROOT/run_falcon3_7b_evals.sh"

usage() { awk 'NR==1{next} /^#/{sub(/^# ?/,"");print;next} {exit}' "$0"; exit "${1:-0}"; }

# Captured up front so a relative --dir resolves where the user ran this
# script, not somewhere the child script might cd to.
INVOKE_PWD="$PWD"

LOOPS="1"; LOOP_DIR=""; ITER_TIMEOUT=""
PORT="8019"; SERVER_URL=""; OUTPUT_SET=0
PASSTHROUGH=()

while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help) usage 0 ;;
    --loops|--dir|--timeout)
      key="$1"; val="${2:-}"; shift 2 || { echo "ERROR: $key needs a value"; exit 1; } ;;
    --loops=*|--dir=*|--timeout=*)
      key="${1%%=*}"; val="${1#*=}"; shift ;;
    --port|--server-url|--output|--tasks|--limit|--ifeval-limit|--gpqa-limit|--concurrent|--model)
      key="$1"; val="${2:-}"; PASSTHROUGH+=("$1" "$val"); shift 2 || { echo "ERROR: $key needs a value"; exit 1; } ;;
    --port=*|--server-url=*|--output=*|--tasks=*|--limit=*|--ifeval-limit=*|--gpqa-limit=*|--concurrent=*|--model=*)
      key="${1%%=*}"; val="${1#*=}"; PASSTHROUGH+=("$1"); shift ;;
    *) echo "ERROR: unknown arg '$1'"; usage 1 ;;
  esac
  case "$key" in
    --loops) LOOPS="$val" ;;
    --dir) LOOP_DIR="$val" ;;
    --timeout) ITER_TIMEOUT="$val" ;;
    --port) PORT="$val" ;;
    --server-url) SERVER_URL="$val" ;;
    --output) OUTPUT_SET=1 ;;
  esac
done

case "$LOOPS" in ''|*[!0-9]*) echo "ERROR: --loops must be a positive integer, got '$LOOPS'"; exit 1 ;; esac
[ "$LOOPS" -ge 1 ] || { echo "ERROR: --loops must be >= 1"; exit 1; }
if [ -n "$ITER_TIMEOUT" ]; then
  case "$ITER_TIMEOUT" in ''|*[!0-9]*) echo "ERROR: --timeout must be a positive integer (seconds), got '$ITER_TIMEOUT'"; exit 1 ;; esac
  [ "$ITER_TIMEOUT" -ge 1 ] || { echo "ERROR: --timeout must be >= 1"; exit 1; }
fi
if [ -n "$LOOP_DIR" ]; then
  case "$LOOP_DIR" in /*) ;; *) LOOP_DIR="$INVOKE_PWD/$LOOP_DIR" ;; esac
  mkdir -p "$LOOP_DIR"
fi

HOST_FOR_CHECK="${SERVER_URL:-http://127.0.0.1}"

# Progress lines go to stderr so they stay visible/interleaved correctly in the
# master log even when an iteration's own output is redirected to a file.
ts() { date '+%Y-%m-%d %H:%M:%S'; }
hms() { printf '%dm%02ds' $(( $1 / 60 )) $(( $1 % 60 )); }
say() { echo "$*" >&2; }

# Single run, no per-iteration dir: behave exactly like calling the eval
# script directly.
if [ "$LOOPS" = "1" ] && [ -z "$LOOP_DIR" ] && [ -z "$ITER_TIMEOUT" ]; then
  exec "$EVAL_SCRIPT" "${PASSTHROUGH[@]}"
fi

say "[$(ts)] loop start: loops=$LOOPS port=$PORT${LOOP_DIR:+ dir=$LOOP_DIR}${ITER_TIMEOUT:+ timeout=${ITER_TIMEOUT}s}"
[ -z "$ITER_TIMEOUT" ] && say "[$(ts)] no --timeout: a hung iteration will block here indefinitely (by design — attach gdb/py-spy to the EngineCore then)."

loop_start=$(date +%s)
passed=0; failed=0; timed_out=0; last_iter=0
width=${#LOOPS}

for i in $(seq 1 "$LOOPS"); do
  last_iter="$i"
  iter_tag=$(printf "%0${width}d" "$i")
  if [ -n "$LOOP_DIR" ]; then
    iter_log="$LOOP_DIR/iter_${iter_tag}.log"
  else
    iter_log=""
  fi

  iter_args=("${PASSTHROUGH[@]}")
  if [ -n "$LOOP_DIR" ] && [ "$OUTPUT_SET" = "0" ]; then
    iter_args+=(--output "$LOOP_DIR/iter_${iter_tag}_results")
  fi

  # Cheap liveness probe. A #4521-style wedge still answers /v1/models
  # (uvicorn is alive, only generation is stuck), so this does not
  # false-positive on a hang -- it only catches a server that actually died.
  if ! curl -sf "${HOST_FOR_CHECK}:${PORT}/health" >/dev/null 2>&1 \
     && ! curl -sf "${HOST_FOR_CHECK}:${PORT}/v1/models" >/dev/null 2>&1; then
    say "[$(ts)] iter ${i}/${LOOPS} ABORT   server at ${HOST_FOR_CHECK}:${PORT} is not answering — stopping loop"
    break
  fi

  say "[$(ts)] iter ${i}/${LOOPS} START${iter_log:+   log=$iter_log}"
  iter_start=$(date +%s)

  rc=0
  if [ -n "$ITER_TIMEOUT" ]; then
    # --foreground so the timeout applies to this non-interactive child; -k
    # follows with SIGKILL 30s later if the eval script ignores the TERM.
    if [ -n "$iter_log" ]; then
      timeout --foreground -k 30 "$ITER_TIMEOUT" "$EVAL_SCRIPT" "${iter_args[@]}" >"$iter_log" 2>&1 || rc=$?
    else
      timeout --foreground -k 30 "$ITER_TIMEOUT" "$EVAL_SCRIPT" "${iter_args[@]}" || rc=$?
    fi
  else
    if [ -n "$iter_log" ]; then
      "$EVAL_SCRIPT" "${iter_args[@]}" >"$iter_log" 2>&1 || rc=$?
    else
      "$EVAL_SCRIPT" "${iter_args[@]}" || rc=$?
    fi
  fi

  elapsed=$(( $(date +%s) - iter_start ))
  if [ -n "$ITER_TIMEOUT" ] && { [ "$rc" = "124" ] || [ "$rc" = "137" ]; }; then
    timed_out=$(( timed_out + 1 ))
    say "[$(ts)] iter ${i}/${LOOPS} TIMEOUT after ${elapsed}s ($(hms $elapsed)) — exceeded --timeout ${ITER_TIMEOUT}s, stopping loop"
    say "[$(ts)] NOTE: the eval script was killed, which disconnects the eval client; that disconnect may have already cleared a wedged server."
    break
  elif [ "$rc" = "0" ]; then
    passed=$(( passed + 1 ))
    say "[$(ts)] iter ${i}/${LOOPS} DONE    rc=0 elapsed=${elapsed}s ($(hms $elapsed))"
  else
    failed=$(( failed + 1 ))
    say "[$(ts)] iter ${i}/${LOOPS} FAIL    rc=${rc} elapsed=${elapsed}s ($(hms $elapsed))${iter_log:+ — see $iter_log}"
  fi
done

total=$(( $(date +%s) - loop_start ))
say "[$(ts)] loop end: ${passed} passed, ${failed} failed, ${timed_out} timed out, of ${last_iter}/${LOOPS} started — total $(hms $total)"
[ -n "$LOOP_DIR" ] && say "[$(ts)] per-iteration logs: $LOOP_DIR"

# Non-zero if anything went wrong, so a wrapping script can notice.
[ "$failed" = "0" ] && [ "$timed_out" = "0" ]
