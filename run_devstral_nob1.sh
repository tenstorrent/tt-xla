#!/usr/bin/env bash
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
cd /home/ssalice/temp/tt-xla
source venv/activate >/dev/null 2>&1
LOGDIR=/home/ssalice/temp/tt-xla/logs/4k_mixed
LOG="$LOGDIR/devstral_4k_b8_nob1.log"
echo "=== [$(date -u +%H:%M:%S)] START devstral pbt=0 (b1 serial prefill DISABLED) ===" | tee -a "$LOGDIR/driver.log"
timeout 14400 python -m pytest -svv \
  "tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py::test_dptp_devstral_mixed_4k[mesh_shape0-4096-8-0]" \
  --log-memory > "$LOG" 2>&1
rc=$?
echo "=== [$(date -u +%H:%M:%S)] END devstral pbt=0 rc=$rc ===" | tee -a "$LOGDIR/driver.log"
echo "devstral_4k_b8_nob1 rc=$rc" >> "$LOGDIR/rc.txt"
grep -aE "^\[ *[0-9]+\] (MULTI|single)" "$LOG"
grep -aE "empty of|PASSED|FAILED" "$LOG" | tail -3
