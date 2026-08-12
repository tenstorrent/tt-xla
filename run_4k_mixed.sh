#!/usr/bin/env bash
# Driver for the 4k mixed chunked/non-chunked DP+TP runs.
# Runs inside the tt-xla container. Logs everything; never dies on one failure.
# NOTE: no `set -u` -- venv/activate dereferences an unbound LD_LIBRARY_PATH
# on line 10, which kills the shell instantly under nounset.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
cd /home/ssalice/temp/tt-xla
source venv/activate >/dev/null 2>&1

LOGDIR=/home/ssalice/temp/tt-xla/logs/4k_mixed
mkdir -p "$LOGDIR"
TESTFILE=tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py

run_one () {
  local name="$1" nodeid="$2" budget="$3"
  local log="$LOGDIR/${name}.log"
  echo "=== [$(date -u +%H:%M:%S)] START $name (budget ${budget}s) ===" | tee -a "$LOGDIR/driver.log"
  timeout "$budget" python -m pytest -svv "${TESTFILE}::${nodeid}" \
      --log-memory > "$log" 2>&1
  local rc=$?
  echo "=== [$(date -u +%H:%M:%S)] END $name rc=$rc ===" | tee -a "$LOGDIR/driver.log"
  echo "$name rc=$rc" >> "$LOGDIR/rc.txt"
  # quick triage line so the summary is readable without opening the log
  {
    echo "--- $name (rc=$rc) ---"
    grep -E "^\[ *[0-9]+\] (MULTI|single)" "$log" | head -5
    grep -cE "^\[ *[0-9]+\] (MULTI|single)" "$log" | sed 's/^/rows printed: /'
    grep -E "empty of|PASSED|FAILED|ERROR" "$log" | tail -5
    grep -iE "Result shape must match query shape|assert|Error|Exception" "$log" | tail -8
  } >> "$LOGDIR/summary.txt" 2>&1
  return $rc
}

reset_devices () {
  echo "=== [$(date -u +%H:%M:%S)] device reset ===" | tee -a "$LOGDIR/driver.log"
  ( cd /home/ssalice/temp/tt-smi 2>/dev/null && source .venv/bin/activate 2>/dev/null \
      && timeout 600 tt-smi -glx_reset ) >> "$LOGDIR/reset.log" 2>&1
  echo "reset rc=$?" >> "$LOGDIR/driver.log"
}

echo "===== driver start $(date -u) =====" | tee -a "$LOGDIR/driver.log"

run_one qwen3-32b_4k_b32 "test_dptp_qwen_mixed_4k[mesh_shape0-4096-32]" 10800
reset_devices
run_one devstral_4k_b8  "test_dptp_devstral_mixed_4k[mesh_shape0-4096-8]" 14400

echo "===== driver done $(date -u) =====" | tee -a "$LOGDIR/driver.log"
cat "$LOGDIR/rc.txt"
