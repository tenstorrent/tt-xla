#!/bin/bash
# Run each sampling overhead test individually, simplest to most complex.
# Logs go to perf_debug/sampling_overhead_logs/
DIR=perf_debug/sampling_overhead_logs
mkdir -p $DIR

tt-smi -r 0
TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py standalone |& tee $DIR/1_standalone.log

tt-smi -r 0
TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py greedy |& tee $DIR/2_greedy.log

tt-smi -r 0
TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py topk_only |& tee $DIR/3_topk_only.log

tt-smi -r 0
TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py topk_pad |& tee $DIR/4_topk_pad.log

tt-smi -r 0
TT_USE_TTNN_SAMPLING=1 python3 perf_debug/test_sampling_op_overhead.py sampling |& tee $DIR/5_sampling.log

echo ""
echo "=== Results ==="
grep "ms/call" $DIR/*.log
