#!/bin/bash
# Run each sampling overhead test individually, simplest to most complex.
# Logs go to perf_debug/sampling_overhead_logs_sync/
DIR=perf_debug/sampling_overhead_logs_sync
mkdir -p $DIR

TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py standalone |& tee $DIR/1_standalone.log
TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py greedy     |& tee $DIR/2_greedy.log
TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py topk_only  |& tee $DIR/3_topk_only.log
TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py topk_pad   |& tee $DIR/4_topk_pad.log
TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py sampling   |& tee $DIR/5_sampling.log
TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py sampling_b32 |& tee $DIR/5_sampling_b32.log

echo ""
echo "=== Results ==="
grep "ms/call" $DIR/*.log
