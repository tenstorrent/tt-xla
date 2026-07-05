#!/usr/bin/env bash
# Chunked-prefill qualification sweep for the refactored tt-mlir chunked SDPA op.
# Sweeps llama-3.2-3b, llama-3.1-8b, qwen3-8b, qwen3-4b and falcon3-7b-base at
# max_model_len 128 and 64K (b32, device sampling, trace, BFP8 weights+KV), full model
# (num_hidden_layers hack commented out below), with prefill request-count bucketing
# (min_num_seqs=1, prefill_batch_threshold=16; tt-xla #5363). max_num_batched_tokens =
# batch x prefill_chunk (= 65536) caps the prefill bucket to one chunk. The 64K target
# is capped per model to its max context (Qwen3 -> 40960, Falcon3-7B-Base -> 32768).
# Runs are sequential (one device, exclusive). Each writes its own log; a summary
# table is printed and saved at the end.

set -u

cd "$(dirname "$0")"

# ---- knobs ---------------------------------------------------------------
OPT_LEVEL=1
TRACE=1
CPU_SAMPLING=0          # device sampling
KV_CACHE_DTYPE=bfp_bf8  # BFP8 KV (weights BFP8 by default)
BATCH=32
# Per-model chunk + gmu: the shipping chunk-1024 production config (cnn.yaml /
# FINDINGS qualification matrix). All models at chunk 1024; gmu 0.5 for the 3B/4B,
# 0.35 for the 7-8B.
declare -A MODEL_CHUNK=(
  ["llama-3.2-3b"]=1024
  ["llama-3.1-8b"]=1024
  ["qwen3-8b"]=1024
  ["qwen3-4b"]=1024
  ["falcon3-7b-base"]=1024
)
declare -A MODEL_GMU=(
  ["llama-3.2-3b"]=0.5
  ["llama-3.1-8b"]=0.35
  ["qwen3-8b"]=0.35
  ["qwen3-4b"]=0.5
  ["falcon3-7b-base"]=0.35
)
# NUM_HIDDEN_LAYERS=1   # single-layer quick-compile hack; leave commented for full-model sweep
MIN_NUM_SEQS=1          # prefill request-count bucketing (tt-xla #5363)
PREFILL_BATCH_THRESHOLD=16   # small-prefill scheduling threshold (tt-xla #5363)

# Default model set; override with e.g. SWEEP_MODELS="qwen3-8b falcon3-7b-base" to run a subset.
DEFAULT_MODELS=("llama-3.2-3b" "llama-3.1-8b" "qwen3-8b" "qwen3-4b" "falcon3-7b-base")
if [ -n "${SWEEP_MODELS:-}" ]; then read -ra MODELS <<< "$SWEEP_MODELS"; else MODELS=("${DEFAULT_MODELS[@]}"); fi
SEQ_LENS=(65536)   # big length only; capped per model (qwen3 -> 40960, falcon3-7b -> 32768)

# Per-model max context: the 64K target is capped to each model's limit (Qwen3
# tops out at 40960, Falcon3-7B-Base at 32768). A seq_len that collapses onto an
# already-run effective length after capping is skipped.
declare -A MODEL_MAX_CTX=(
  ["llama-3.2-3b"]=131072
  ["llama-3.1-8b"]=131072
  ["qwen3-8b"]=40960
  ["qwen3-4b"]=40960
  ["falcon3-7b-base"]=32768
)
# --------------------------------------------------------------------------

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="chunked_prefill_qual_${TS}"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/SUMMARY.md"

echo "# Chunked-prefill qualification sweep ($TS)" | tee "$SUMMARY"
{
  echo
  echo "Settings: opt=$OPT_LEVEL trace=$TRACE cpu_sampling=$CPU_SAMPLING kv=$KV_CACHE_DTYPE"
  echo "batch=$BATCH (per-model chunk+gmu below); max_num_batched_tokens = batch * chunk"
  echo "num_hidden_layers=${NUM_HIDDEN_LAYERS:-full} min_num_seqs=$MIN_NUM_SEQS prefill_batch_threshold=$PREFILL_BATCH_THRESHOLD"
  echo
  echo "| model | seq_len | chunk | gmu | result | wall | samples/s | TTFT (ms) | note |"
  echo "|---|---|---|---|---|---|---|---|---|"
} | tee -a "$SUMMARY"

for MODEL in "${MODELS[@]}"; do
  MAXCTX=${MODEL_MAX_CTX[$MODEL]:-65536}
  CHUNK=${MODEL_CHUNK[$MODEL]:-2048}
  GMU=${MODEL_GMU[$MODEL]:-0.325}
  MAX_NUM_BATCHED_TOKENS=$(( BATCH * CHUNK ))
  declare -A _seen_seqs=()
  for SEQ in "${SEQ_LENS[@]}"; do
    # Cap the requested seq_len to the model's max context, then skip if a prior
    # (smaller) requested seq already covered this effective length.
    EFFSEQ=$SEQ
    [ "$SEQ" -gt "$MAXCTX" ] && EFFSEQ=$MAXCTX
    if [ -n "${_seen_seqs[$EFFSEQ]:-}" ]; then
      echo ">>> [$(date +%H:%M:%S)] $MODEL seq_len=$SEQ capped to $EFFSEQ (already run) -- skip"
      continue
    fi
    _seen_seqs[$EFFSEQ]=1
    TAG="${MODEL//./}_${EFFSEQ}"
    LOG="$LOGDIR/${TAG}.log"
    JSON="$LOGDIR/${TAG}.json"
    echo ">>> [$(date +%H:%M:%S)] $MODEL seq_len=$EFFSEQ -> $LOG"

    # Only pass num_hidden_layers when the quick-compile hack is enabled above.
    NHL_ENV=()
    [ -n "${NUM_HIDDEN_LAYERS:-}" ] && NHL_ENV=("TT_BENCHMARK_NUM_HIDDEN_LAYERS=$NUM_HIDDEN_LAYERS")

    env "${NHL_ENV[@]}" \
    _BENCH_OPTIMIZATION_LEVEL=$OPT_LEVEL \
    TT_BENCHMARK_TRACE=$TRACE \
    TT_BENCHMARK_CPU_SAMPLING=$CPU_SAMPLING \
    TT_BENCHMARK_KV_CACHE_DTYPE=$KV_CACHE_DTYPE \
    TT_BENCHMARK_BATCH_SIZE=$BATCH \
    TT_BENCHMARK_PREFILL_CHUNK_SIZE=$CHUNK \
    TT_BENCHMARK_GMU=$GMU \
    TT_BENCHMARK_MAX_MODEL_LEN=$EFFSEQ \
    TT_BENCHMARK_MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS \
    TT_BENCHMARK_MIN_NUM_SEQS=$MIN_NUM_SEQS \
    TT_BENCHMARK_PREFILL_BATCH_THRESHOLD=$PREFILL_BATCH_THRESHOLD \
    python -m pytest -svv tests/benchmark/test_vllm_benchmarks.py::test_vllm_benchmark \
      -k "$MODEL" --output-file "$JSON" > "$LOG" 2>&1
    RC=$?

    # parse results
    WALL=$(grep -oE "in [0-9.]+s \(0:[0-9:]+\)" "$LOG" | tail -1 | grep -oE "0:[0-9:]+")
    SPS=$(grep -iE "Avg. samples per second" "$LOG" | tail -1 | grep -oE "[0-9.]+" | tail -1)
    TTFT=$(grep -iE "^\| TTFT \(ms\)" "$LOG" | tail -1 | grep -oE "[0-9.]+" | tail -1)
    if grep -qE "[0-9]+ passed" "$LOG"; then
      RESULT="PASS"
    else
      RESULT="FAIL"
    fi
    NOTE=""
    if grep -qiE "DataType mismatch|emitOpError|failed to legalize" "$LOG"; then NOTE="verifier/legalize"; fi
    if grep -qiE "out of memory|OOM|Out of Memory|alloc.*fail|warmup" "$LOG" && [ "$RESULT" = "FAIL" ]; then NOTE="${NOTE:+$NOTE,}possible-OOM"; fi
    if grep -qiE "Fatal error|FATAL" "$LOG"; then NOTE="${NOTE:+$NOTE,}fatal"; fi

    echo "| $MODEL | $EFFSEQ | $CHUNK | $GMU | $RESULT | ${WALL:-?} | ${SPS:-?} | ${TTFT:-?} | ${NOTE:-} |" | tee -a "$SUMMARY"
  done
done

echo | tee -a "$SUMMARY"
echo "Done. Logs + summary in: $LOGDIR" | tee -a "$SUMMARY"
