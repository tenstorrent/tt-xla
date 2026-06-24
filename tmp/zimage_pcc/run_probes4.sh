#!/bin/bash
# Round 4: pinpoint adaLN vs embedders for the minimal fix.
cd /proj_sw/user_dev/ctr-akannan/24_2_jun_yyz/tt-xla
source venv/activate 2>/dev/null
export ZIMAGE_NLAYERS="${ZIMAGE_NLAYERS:-8}"
OUT=tmp/zimage_pcc
SUM=$OUT/summary.txt
echo "=== probe run4 NLAYERS=$ZIMAGE_NLAYERS $(date) ===" >> $SUM
TESTS="repl_adaln_only repl_emb_only"
for t in $TESTS; do
  log=$OUT/n${ZIMAGE_NLAYERS}_${t}.log
  echo ">>> running $t (NLAYERS=$ZIMAGE_NLAYERS) -> $log"
  pytest -svv "tests/torch/models/z_image/test_pcc_probe.py::test_pcc_${t}" \
      -p no:cacheprovider > "$log" 2>&1
  rc=$?
  line=$(grep -a "PCC-PROBE" "$log" | tail -1)
  if [ -z "$line" ]; then
    line="NO-PCC (rc=$rc) $(grep -aE 'Segmentation|core dumped|RuntimeError|Error code|error:|number of output' "$log" | grep -aivE 'Wno-error|Werror' | tail -1)"
  fi
  echo "n${ZIMAGE_NLAYERS}_${t}: $line" | tee -a $SUM
done
echo "=== done4 $(date) ===" >> $SUM
