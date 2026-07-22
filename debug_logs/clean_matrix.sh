#!/bin/bash
# Host-side: reset device + wait + run one config in container, per entry. Reliable (clean device each).
RESET=/home/mvasiljevic/.ttsmi-venv/bin/tt-smi
run_one() {
  local name="$1"; shift
  local envs="$1"; shift
  docker exec --user 6002:6002 mvasiljevic-ttxla bash -lc 'for p in $(pgrep -f test_llms); do kill -9 $p 2>/dev/null; done; sleep 2' >/dev/null 2>&1
  $RESET -r >/dev/null 2>&1
  sleep 15
  docker exec --user 6002:6002 --workdir /home/mvasiljevic/tt-xla mvasiljevic-ttxla bash -lc "
    source venv/activate >/dev/null 2>&1
    env $envs timeout 400 ./run_qb2_experiment.sh $name > /home/mvasiljevic/cm_$name.log 2>&1"
  echo "===== $name ($envs) ====="
  grep -iE 'First decode PCC=|First decode PCC raised|Prefill PCC=|eager decode PCC|COMPILED_DUMP\] logits|PASSED|FAILED' /home/mvasiljevic/cm_$name.log 2>/dev/null | grep -viE 'nanobind' | head -6
  echo
}
run_one cm_baseline "QB2_MESH=2d QB2_OPT=1"
run_one cm_traceoff "QB2_MESH=2d QB2_OPT=1 QB2_TRACE=0"
run_one cm_eager    "QB2_MESH=2d QB2_OPT=1 QB2_EAGER_DUMP=1"
run_one cm_keepalive "QB2_MESH=2d QB2_OPT=1 QB2_COMPILED_DUMP=1"
echo "===== CLEAN MATRIX DONE ====="
