#!/usr/bin/env bash
# Chunked-prefill qualification sweep for the refactored tt-mlir chunked SDPA op.
# Sweeps llama-3.2-3b and llama-3.1-8b at 128 / 4K / 32K / 64K with the best-known
# vLLM benchmark settings, plus two experiments vs the prior known-good config:
#   - gmu raised 0.15 -> 0.35
#   - max_num_batched_tokens set to batch x prefill_chunk (= 65536) instead of the 2M default
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
PREFILL_CHUNK=2048
GMU=0.35                # experiment: was 0.15 @ 64K
MAX_NUM_BATCHED_TOKENS=$(( BATCH * PREFILL_CHUNK ))   # = 65536, vs 2M default

MODELS=("llama-3.2-3b" "llama-3.1-8b")
SEQ_LENS=(128 4096 32768 65536)
# --------------------------------------------------------------------------

TS=$(date +%Y%m%d_%H%M%S)
LOGDIR="chunked_prefill_qual_${TS}"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/SUMMARY.md"

echo "# Chunked-prefill qualification sweep ($TS)" | tee "$SUMMARY"
{
  echo
  echo "Settings: opt=$OPT_LEVEL trace=$TRACE cpu_sampling=$CPU_SAMPLING kv=$KV_CACHE_DTYPE"
  echo "batch=$BATCH prefill_chunk=$PREFILL_CHUNK gmu=$GMU max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
  echo
  echo "| model | seq_len | result | wall | samples/s | TTFT (ms) | note |"
  echo "|---|---|---|---|---|---|---|"
} | tee -a "$SUMMARY"

for MODEL in "${MODELS[@]}"; do
  for SEQ in "${SEQ_LENS[@]}"; do
    TAG="${MODEL//./}_${SEQ}"
    LOG="$LOGDIR/${TAG}.log"
    JSON="$LOGDIR/${TAG}.json"
    echo ">>> [$(date +%H:%M:%S)] $MODEL seq_len=$SEQ -> $LOG"

    _BENCH_OPTIMIZATION_LEVEL=$OPT_LEVEL \
    TT_BENCHMARK_TRACE=$TRACE \
    TT_BENCHMARK_CPU_SAMPLING=$CPU_SAMPLING \
    TT_BENCHMARK_KV_CACHE_DTYPE=$KV_CACHE_DTYPE \
    TT_BENCHMARK_BATCH_SIZE=$BATCH \
    TT_BENCHMARK_PREFILL_CHUNK_SIZE=$PREFILL_CHUNK \
    TT_BENCHMARK_GMU=$GMU \
    TT_BENCHMARK_MAX_MODEL_LEN=$SEQ \
    TT_BENCHMARK_MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS \
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

    echo "| $MODEL | $SEQ | $RESULT | ${WALL:-?} | ${SPS:-?} | ${TTFT:-?} | ${NOTE:-} |" | tee -a "$SUMMARY"
  done
done

echo | tee -a "$SUMMARY"
echo "Done. Logs + summary in: $LOGDIR" | tee -a "$SUMMARY"
