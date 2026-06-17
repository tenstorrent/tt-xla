#!/usr/bin/env bash
# bf16 PCC depth/block-type sweep on the 4-chip Blackhole QB, (1,4) mesh.
# Goal: distinguish accumulative-across-layers vs single-block/op PCC drop.
#   Dual-only ladder (NS=0):      NL = 1, 4, 8
#   Single-block ladder (NL=8):   NS = 0, 12, 24, 36, 48
# Each config -> its own log; PCC scraped into the summary file.
set -uo pipefail
cd /localdev/ctr-akannan/17_jun_sjc/tt-xla

export HF_HOME=/localdev/ctr-akannan/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TT_METAL_CACHE=/localdev/ctr-akannan/.cache/tt_metal
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export HF_TOKEN

OUT=flux_updated_logs/pcc_sweep
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"
: > "$SUMMARY"

# config list: "NL NS"
configs=( "1 0" "4 0" "8 0" "8 12" "8 24" "8 36" "8 48" )

for c in "${configs[@]}"; do
  set -- $c; NL=$1; NS=$2
  tag="nl${NL}_ns${NS}"
  log="$OUT/${tag}.log"
  echo "=== $(date '+%H:%M:%S') START NL=$NL NS=$NS -> $log ===" | tee -a "$SUMMARY"
  FLUX_NL=$NL FLUX_NS=$NS FLUX_SHARDED=1 \
    timeout 3600 python -m pytest -svv \
    tests/torch/models/flux2/test_transformer_realwt_isolate.py::test_realwt \
    > "$log" 2>&1
  rc=$?
  pcc=$(grep -aoE "PCC = [-0-9.]+" "$log" | tail -1)
  res=$(grep -aoE "[0-9]+ (passed|failed|error)" "$log" | tail -1)
  echo "    NL=$NL NS=$NS  rc=$rc  ${pcc:-PCC=NONE}  [${res:-no-result}]" | tee -a "$SUMMARY"
done

echo "=== $(date '+%H:%M:%S') SWEEP DONE ===" | tee -a "$SUMMARY"
echo "SWEEP_EXIT=0"
