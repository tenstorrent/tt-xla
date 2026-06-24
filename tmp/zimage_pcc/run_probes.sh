#!/bin/bash
# Run each sharded probe config in its OWN pytest process (avoids the
# multi-config-in-one-process segfault in _XLAC). Amplify per-layer error
# with ZIMAGE_NLAYERS. Collect PCC-PROBE lines into summary.txt.
cd /proj_sw/user_dev/ctr-akannan/24_2_jun_yyz/tt-xla
source venv/activate 2>/dev/null
export ZIMAGE_NLAYERS="${ZIMAGE_NLAYERS:-8}"
OUT=tmp/zimage_pcc
SUM=$OUT/summary.txt
echo "=== probe run NLAYERS=$ZIMAGE_NLAYERS $(date) ===" >> $SUM

# key discriminators first
TESTS="model4 no_reduce model4_fp32 no_attn_reduce no_ffn_reduce model2"
for t in $TESTS; do
  log=$OUT/n${ZIMAGE_NLAYERS}_${t}.log
  echo ">>> running $t (NLAYERS=$ZIMAGE_NLAYERS) -> $log"
  pytest -svv "tests/torch/models/z_image/test_pcc_probe.py::test_pcc_${t}" \
      -p no:cacheprovider > "$log" 2>&1
  rc=$?
  line=$(grep -a "PCC-PROBE" "$log" | tail -1)
  if [ -z "$line" ]; then
    # capture crash / failure signature
    line="NO-PCC (rc=$rc) $(grep -aE 'Segmentation|core dumped|RuntimeError|Error:|assert' "$log" | grep -aivE 'Wno-error|Werror' | tail -1)"
  fi
  echo "n${ZIMAGE_NLAYERS}_${t}: $line" | tee -a $SUM
done
echo "=== done $(date) ===" >> $SUM
