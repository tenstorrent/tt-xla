#!/bin/bash
# Validate the #5738 rectangular-grid fix: reset device, then run the 2D-mesh
# decode 3x (run 1 = fresh device, runs 2-3 = reused) to check the decode PCC
# recovers AND is deterministic (the bug was nondeterministic across runs).
RESET=/home/mvasiljevic/.ttsmi-venv/bin/tt-smi
docker exec --user 6002:6002 mvasiljevic-ttxla bash -lc 'for p in $(pgrep -f test_llms); do kill -9 $p 2>/dev/null; done; sleep 2' >/dev/null 2>&1
$RESET -r >/dev/null 2>&1
sleep 20
for run in 1 2 3; do
  docker exec --user 6002:6002 --workdir /home/mvasiljevic/tt-xla mvasiljevic-ttxla bash -lc "
    source venv/activate >/dev/null 2>&1
    QB2_MESH=2d QB2_OPT=1 timeout 400 ./run_qb2_experiment.sh fixval_$run > /home/mvasiljevic/fixval_$run.log 2>&1"
  echo "----- RUN $run (reset only before run 1) -----"
  grep -iE "First decode PCC=|Prefill PCC=|decode PCC=|PASSED|FAILED" /home/mvasiljevic/fixval_$run.log 2>/dev/null | grep -viE "nanobind" | head -6
done
echo "===== VALIDATION DONE ====="
