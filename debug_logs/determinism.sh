#!/bin/bash
docker exec --user 6002:6002 mvasiljevic-ttxla bash -lc 'for p in $(pgrep -f test_llms); do kill -9 $p 2>/dev/null; done; sleep 2' >/dev/null 2>&1
/home/mvasiljevic/.ttsmi-venv/bin/tt-smi -r >/dev/null 2>&1
sleep 15
for run in 1 2 3; do
  docker exec --user 6002:6002 --workdir /home/mvasiljevic/tt-xla mvasiljevic-ttxla bash -lc "
    source venv/activate >/dev/null 2>&1
    QB2_MESH=2d QB2_OPT=1 timeout 300 ./run_qb2_experiment.sh det_run$run > /home/mvasiljevic/det_run$run.log 2>&1"
  echo "----- RUN $run (reset only before run 1) -----"
  grep -iE "First decode PCC=|Prefill PCC=" /home/mvasiljevic/det_run$run.log 2>/dev/null | grep -viE "nanobind" | head -2
done
echo "===== DETERMINISM DONE ====="
