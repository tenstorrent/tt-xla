#!/usr/bin/env bash
# Chunked-prefill qualification sweep for the refactored tt-mlir chunked SDPA op.
# Sweeps llama-3.2-3b, llama-3.1-8b, qwen3-8b, qwen3-4b and falcon3-7b-base at
# 128 / 4K / 32K / 64K (b32, device sampling, trace, BFP8 weights+KV) with the
# best-known vLLM benchmark settings. max_num_batched_tokens = batch x prefill_chunk
# (= 65536) caps the prefill bucket to one chunk. The 64K target is capped per
# model to its max context (Qwen3 -> 40960, Falcon3-7B-Base -> 32768).
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
GMU=0.15                # known-good (b32 @64K). 0.35/0.325/0.30 OOM llama-3.1-8b: the
                        # [batch x chunk x intermediate] prefill activation needs free DRAM,
                        # so KV budget must stay low. 0.15 is the min that still inits 64K KV.
MAX_NUM_BATCHED_TOKENS=$(( BATCH * PREFILL_CHUNK ))   # = 65536, vs 2M default

# Default model set; override with e.g. SWEEP_MODELS="qwen3-8b falcon3-7b-base" to run a subset.
DEFAULT_MODELS=("llama-3.2-3b" "llama-3.1-8b" "qwen3-8b" "qwen3-4b" "falcon3-7b-base")
if [ -n "${SWEEP_MODELS:-}" ]; then read -ra MODELS <<< "$SWEEP_MODELS"; else MODELS=("${DEFAULT_MODELS[@]}"); fi
SEQ_LENS=(128 4096 32768 65536)

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
  echo "batch=$BATCH prefill_chunk=$PREFILL_CHUNK gmu=$GMU max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS"
  echo
  echo "| model | seq_len | result | wall | samples/s | TTFT (ms) | note |"
  echo "|---|---|---|---|---|---|---|"
} | tee -a "$SUMMARY"

for MODEL in "${MODELS[@]}"; do
  MAXCTX=${MODEL_MAX_CTX[$MODEL]:-65536}
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

    _BENCH_OPTIMIZATION_LEVEL=$OPT_LEVEL \
    TT_BENCHMARK_TRACE=$TRACE \
    TT_BENCHMARK_CPU_SAMPLING=$CPU_SAMPLING \
    TT_BENCHMARK_KV_CACHE_DTYPE=$KV_CACHE_DTYPE \
    TT_BENCHMARK_BATCH_SIZE=$BATCH \
    TT_BENCHMARK_PREFILL_CHUNK_SIZE=$PREFILL_CHUNK \
    TT_BENCHMARK_GMU=$GMU \
    TT_BENCHMARK_MAX_MODEL_LEN=$EFFSEQ \
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

    echo "| $MODEL | $EFFSEQ | $RESULT | ${WALL:-?} | ${SPS:-?} | ${TTFT:-?} | ${NOTE:-} |" | tee -a "$SUMMARY"
  done
done

echo | tee -a "$SUMMARY"
echo "Done. Logs + summary in: $LOGDIR" | tee -a "$SUMMARY"
