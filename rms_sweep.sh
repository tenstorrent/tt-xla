#!/bin/bash
cd /home/mvasiljevic/tt-xla
source venv/activate >/dev/null 2>&1
run() {
  echo "===== $1 hidden=$2 batch=$3 grid=$4x$5 ====="
  RMS_HIDDEN=$2 RMS_BATCH=$3 RMS_GX=$4 RMS_GY=$5 timeout 200 python dist_rmsnorm_repro.py 2>&1 | grep -iE 'HIDDEN=|PCC\(|dev  max|Error:|FATAL|intersection|Timeout|assert|num_intersections' | head -5
  echo
}
run B_hidden8192 8192 8 3 6
run C_batch32    7168 32 3 6
run D_model_hb   8192 32 3 6
run E_biggrid_bw2 8192 32 7 7   # 8x8=64 cores -> block_w toward model's 2
echo "===== SWEEP DONE ====="
