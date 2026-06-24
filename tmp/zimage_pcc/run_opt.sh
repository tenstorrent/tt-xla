cd /proj_sw/user_dev/ctr-akannan/24_2_jun_yyz/tt-xla
source venv/activate 2>/dev/null
export ZIMAGE_NLAYERS=30
SUM=tmp/zimage_pcc/summary.txt
echo "=== opt-level 30-layer $(date) ===" >> $SUM
for t in model4_opt2 model4_opt1; do
  log=tmp/zimage_pcc/depth30_${t}.log
  echo ">>> $t" 
  pytest -svv "tests/torch/models/z_image/test_pcc_probe.py::test_pcc_${t}" -p no:cacheprovider > "$log" 2>&1
  rc=$?
  line=$(grep -a "PCC-PROBE" "$log" | tail -1)
  [ -z "$line" ] && line="NO-PCC (rc=$rc) $(grep -aE 'exceeds per-core|Error code|number of output|RuntimeError|TT_FATAL' "$log" | grep -aivE 'Wno-error|Werror' | tail -1)"
  echo "depth30_${t}: $line" | tee -a $SUM
done
echo "=== opt done $(date) ===" >> $SUM
