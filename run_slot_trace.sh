#!/usr/bin/env bash
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export TTXLA_SLOT_TRACE=1
cd /home/ssalice/temp/tt-xla
source venv/activate >/dev/null 2>&1
LOGDIR=/home/ssalice/temp/tt-xla/logs/4k_mixed
T=tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py
for pbt in 16 0; do
  LOG="$LOGDIR/slot_trace_pbt${pbt}.log"
  echo "=== [$(date -u +%H:%M:%S)] START slot_trace pbt=$pbt ===" | tee -a "$LOGDIR/driver.log"
  timeout 3600 python -m pytest -svv "${T}::test_dptp_slot_trace[mesh_shape0-8-${pbt}]" > "$LOG" 2>&1
  echo "=== [$(date -u +%H:%M:%S)] END slot_trace pbt=$pbt rc=$? ===" | tee -a "$LOGDIR/driver.log"
  echo "--- SLOTTRACE lines (pbt=$pbt) ---"
  grep -ao "SLOTTRACE phase=[^|]*| [^\"]*" "$LOG" | head -30
done
