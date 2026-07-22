#!/bin/bash
# Sequential investigation matrix for issue #5738 (all: 1 layer, --pcc-decode)
cd /home/mvasiljevic/tt-xla
R=./run_qb2_experiment.sh

# 1) Primary repro: restore 2D (2,N//2) mesh, opt 1
QB2_MESH=2d QB2_OPT=1 $R repro_2d_opt1

# 2) opt level 0 on 2D mesh
QB2_MESH=2d QB2_OPT=0 $R repro_2d_opt0

# 3) RMS-norm localization: 2D mesh but norm weights replicated (no distributed_rms_norm)
QB2_MESH=2d QB2_OPT=1 QB2_NORM=replicate $R norm_replicate_2d_opt1

# 4) KV cache batch-sharded on 2D mesh
QB2_MESH=2d QB2_OPT=1 QB2_KV=batch $R kv_batch_2d_opt1

echo "===== MATRIX COMPLETE $(date) ====="
