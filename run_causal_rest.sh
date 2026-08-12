#!/usr/bin/env bash
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export TTXLA_SLOT_TRACE=1
cd /home/ssalice/temp/tt-xla
source venv/activate >/dev/null 2>&1
LOGDIR=/home/ssalice/temp/tt-xla/logs/4k_mixed
T=tests/integrations/vllm_plugin/generative/test_data_tensor_parallel_generation.py
for case in aligned allshort; do
  LOG="$LOGDIR/causal_ab_${case}.log"
  echo "=== [$(date -u +%H:%M:%S)] START causal_ab $case ===" | tee -a "$LOGDIR/driver.log"
  timeout 5400 python -m pytest -svv "${T}::test_dptp_small_model_causal_ab[mesh_shape0-${case}]" > "$LOG" 2>&1
  echo "=== [$(date -u +%H:%M:%S)] END causal_ab $case rc=$? ===" | tee -a "$LOGDIR/driver.log"
  grep -aE "DEGENERATE_ROWS" "$LOG" | tail -1
done
