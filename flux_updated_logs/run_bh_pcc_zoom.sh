#!/usr/bin/env bash
# Fine zoom into the deep single-block collapse (NL=8, NS 36->48), bf16, (1,4).
# Coarse sweep showed 0.952 (NS=36) -> ~0.650 (NS=48): a 0.30 drop in 12 blocks.
# This walks NS in steps of 2 to tell a single-block cliff from smooth-but-steep
# exponential error compounding.
set -uo pipefail
cd /localdev/ctr-akannan/17_jun_sjc/tt-xla

export HF_HOME=/localdev/ctr-akannan/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TT_METAL_CACHE=/localdev/ctr-akannan/.cache/tt_metal
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export HF_TOKEN

OUT=flux_updated_logs/pcc_zoom
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"
: > "$SUMMARY"

for NS in 38 40 42 44 46; do
  NL=8
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

echo "=== $(date '+%H:%M:%S') ZOOM DONE ===" | tee -a "$SUMMARY"
echo "ZOOM_EXIT=0"
