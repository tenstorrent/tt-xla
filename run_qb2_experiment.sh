#!/bin/bash
# Usage: run_qb2_experiment.sh <name> [extra pytest args]
# Env knobs read by the test: QB2_MESH(tp|2d) QB2_OPT(0|1|2) QB2_KV(default|batch|batchonly|replicate) QB2_NORM(shard|replicate)
set -o pipefail
cd /home/mvasiljevic/tt-xla
source venv/activate >/dev/null 2>&1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
NAME="${1:-run}"; shift || true
LOG=/home/mvasiljevic/exp_${NAME}.log
echo "===== EXPERIMENT $NAME  $(date) =====" | tee "$LOG"
echo "QB2_MESH=${QB2_MESH:-tp} QB2_OPT=${QB2_OPT:-2} QB2_KV=${QB2_KV:-default} QB2_NORM=${QB2_NORM:-shard}" | tee -a "$LOG"
python -m pytest -svv \
  "tests/benchmark/test_llms.py::test_llama_3_1_70b_tp_qb2" \
  --num-layers 1 --pcc-decode "$@" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "===== DONE $NAME rc=$rc $(date) =====" | tee -a "$LOG"
echo "--- PCC lines ---" | tee -a "$LOG"
grep -iE "PCC|prefill|decode|FAILED|PASSED|Error|assert" "$LOG" | tail -40
exit $rc
