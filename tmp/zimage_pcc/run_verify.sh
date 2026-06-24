#!/bin/bash
# Verify the adaLN-replication fix end-to-end via the REAL shard spec
# (loader.load_shard_spec -> shard_transformer_specs). Expect model4 ~0.997.
cd /proj_sw/user_dev/ctr-akannan/24_2_jun_yyz/tt-xla
source venv/activate 2>/dev/null
export ZIMAGE_NLAYERS="${ZIMAGE_NLAYERS:-8}"
OUT=tmp/zimage_pcc
SUM=$OUT/summary.txt
echo "=== VERIFY fix NLAYERS=$ZIMAGE_NLAYERS $(date) ===" >> $SUM
TESTS="model4 model2"
for t in $TESTS; do
  log=$OUT/verify_n${ZIMAGE_NLAYERS}_${t}.log
  echo ">>> verify $t (NLAYERS=$ZIMAGE_NLAYERS) -> $log"
  pytest -svv "tests/torch/models/z_image/test_pcc_probe.py::test_pcc_${t}" \
      -p no:cacheprovider > "$log" 2>&1
  rc=$?
  line=$(grep -a "PCC-PROBE" "$log" | tail -1)
  [ -z "$line" ] && line="NO-PCC (rc=$rc) $(grep -aE 'Error code|error:|number of output|RuntimeError' "$log" | grep -aivE 'Wno-error|Werror' | tail -1)"
  echo "VERIFY_n${ZIMAGE_NLAYERS}_${t}: $line" | tee -a $SUM
done
echo "=== verify done $(date) ===" >> $SUM
