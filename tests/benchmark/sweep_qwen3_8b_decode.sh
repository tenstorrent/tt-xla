#!/usr/bin/env bash
# Pure tt-xla repro of the Qwen3-8B decode-rate slowdown seen on the
# tt-inference-server online server. Sweeps OSL at fixed ISL (concurrency 1,
# single decoder layer) and reports decode tok/s per point. If decode tok/s
# falls as OSL grows, the slowdown reproduces in the offline vLLM engine path
# (i.e. in tt-xla, not just the serving/streaming layer).
#
# Settings mirror tt-inference-server/tt-media-server/launch_qwen3_8b.sh
# (opt=1, gmu=0.30, weights/KV bfp8, prefill_chunk_size=2048,
# fp32_dest_acc_en=false, enable_trace, device sampling) + num_hidden_layers=1.
#
# Each (isl, osl) is its own pytest process (fresh engine build).
#
# Usage:
#   ./sweep_qwen3_8b_decode.sh                 # device 0, ISL 128, OSL {128,1024,4096}
#   TT_VISIBLE_DEVICES=2 ./sweep_qwen3_8b_decode.sh
set -u

# tt-xla root is two levels up from tests/benchmark/.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
TTXLA_ROOT="$(pwd)"

# Results subdir (created first so ALL logs land here, even on early failure).
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="tests/benchmark/sweep_results/qwen3_8b_decode_${STAMP}"
mkdir -p "$OUT"
# Tee everything (this script + each pytest) into the subdir so the whole run is
# inspectable in one place, regardless of how the script is invoked.
exec > >(tee -a "${OUT}/run.log") 2>&1

# Activate the tt-xla venv if it isn't already. Its activate script references
# unbound vars (e.g. LD_LIBRARY_PATH) that trip `set -u`, so relax it here.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f venv/activate ]; then
  set +u
  # shellcheck disable=SC1091
  source venv/activate
  set -u
fi

export TT_VISIBLE_DEVICES="${TT_VISIBLE_DEVICES:-0}"
export TT_MESH_GRAPH_DESC_PATH="${TT_MESH_GRAPH_DESC_PATH:-${TTXLA_ROOT}/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/fabric/mesh_graph_descriptors/p150_mesh_graph_descriptor.textproto}"

# ISL fixed at 128; sweep OSL. (Same buckets as the online-server sweep.)
PAIRS=("128 128" "128 1024" "128 4096")

SUMMARY="${OUT}/summary.tsv"
printf "isl\tosl\tstatus\tttft_ms\tdecode_tok_s\tjson\n" > "$SUMMARY"

echo "device=${TT_VISIBLE_DEVICES}  mesh=${TT_MESH_GRAPH_DESC_PATH}"
echo "results dir -> ${TTXLA_ROOT}/${OUT}  (run.log + per-point .log/.json + summary.tsv)"

for pair in "${PAIRS[@]}"; do
  # shellcheck disable=SC2086
  set -- $pair; isl="$1"; osl="$2"
  id="isl${isl}_osl${osl}"
  json="${OUT}/${id}.json"
  log="${OUT}/${id}.log"

  echo "==================================================================="
  echo "RUN: ${id}  (device ${TT_VISIBLE_DEVICES})  -> ${log}"
  echo "==================================================================="

  pytest -svv \
    "tests/benchmark/test_vllm_benchmarks.py::test_vllm_qwen3_8b_decode_sweep[${id}]" \
    --output-file "$json" 2>&1 | tee "$log"
  status="${PIPESTATUS[0]}"

  # Decode tok/s + TTFT: prefer the per-request line printed by _extract_metrics,
  # fall back to "-" on failure.
  decode_tps="$(grep -oE 'decode_tps=[0-9.]+' "$log" | tail -1 | cut -d= -f2)"
  ttft_ms="$(grep -oE 'TTFT=[0-9.]+ms' "$log" | tail -1 | tr -dc '0-9.')"
  [ -z "$decode_tps" ] && decode_tps="-"
  [ -z "$ttft_ms" ] && ttft_ms="-"
  [ "$status" -eq 0 ] && st="PASS" || st="FAIL(exit=${status})"

  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$isl" "$osl" "$st" "$ttft_ms" "$decode_tps" "$json" >> "$SUMMARY"
done

echo
echo "=== Sweep complete ==="
column -t -s$'\t' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo
echo "If decode_tok_s drops as osl grows (128 -> 1024 -> 4096), the slowdown reproduces in tt-xla."
