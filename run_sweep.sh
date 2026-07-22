#!/bin/bash
cd /home/mvasiljevic/tt-xla
R=./run_qb2_experiment.sh
# (4,1): batch-axis only (heads NOT sharded) — isolates batch-axis sharding as trigger
QB2_MESH=4x1 QB2_OPT=1 QB2_PERUSER=1 $R mesh_4x1_opt1
# (2,2) with trace disabled — isolates capture_or_execute_trace
QB2_MESH=2d QB2_OPT=1 QB2_TRACE=0 QB2_PERUSER=1 $R trace_off_2d_opt1
echo "===== SWEEP COMPLETE $(date) ====="
