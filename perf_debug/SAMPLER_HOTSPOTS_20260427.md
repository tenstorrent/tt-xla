# Sampler hotspots — baseline before non-greedy improvements (2026-04-27)

Snapshot of where device time goes during a single non-greedy sampler step,
captured before any non-greedy sampler perf work has landed on this branch.
Two configurations: the **default tt-cloud-console traffic** path
(`top_k=None, top_p=None`) and the **opt-in top_k=3** path that triggers
`apply_top_k_top_p` and its `probs.sort()` over the full vocab.

## Setup

- Branch: `kmabee/vllm_demo_may1.perf_debug` at HEAD `5b7ee410d` (+ WIP perf
  edits not yet committed). Last upstream commit:
  `714f072ee Add vLLM decode tests with prefill length fixed to 1`.
- Capture: `perf_debug/test_sampler_throughput.py` against
  `tests/integrations/vllm_plugin/sampling/fixtures/llama3_2_3b_decode_step1.pt`
  (Llama-3.2-3B logits, vocab=128256, batch=1, non-greedy temp=0.6,
  device sampling, `TT_USE_TTNN_SAMPLING=1`).
- Tracy: `python -m tracy -p -r -o … -m perf_debug.test_sampler_throughput …`.
  `--sync-host-device` was dropped due to a `device->is_initialized()` profiler
  init crash (see `PERF_DEBUG_GUIDE.md` §2). Tracking via per-iter signposts.

## Full-model end-to-end baselines (`test_sampling_quality`)

`Sample per second` from `pytest tests/benchmark/test_vllm_benchmarks.py::test_sampling_quality[...]`,
captured 2026-04-27. These include model forward + KV cache + sampler — i.e.
what production actually sees per token.

Run dirs (if still on disk):
- batch=1: `sampling_quality_baseline_20260427_034605/`
- batch=2: `sampling_quality_baseline_batch2_20260427_044112/`

| Model         | Batch | greedy-device | greedy-cpu | nongreedy-device | nongreedy-cpu |
|---|---|---|---|---|---|
| llama3.2-1b   | 1     | 82.59         | 67.74      | **5.83**         | 66.13         |
| llama3.2-1b   | 2     | 77.90         | 61.62      | **5.81**         | 60.09         |
| llama3.1-8b   | 1     | 22.05         | 20.64      | **4.88**         | 20.36         |
| llama3.1-8b   | 2     | 21.46         | 19.98      | **4.83**         | 19.65         |

Two things jump out:

1. **`nongreedy-device` is the only slow case.** On 1B it's 14× slower than
   any other config (5.8 vs 60–82 tok/s). On 8B it's 4× slower
   (4.8 vs 19–22 tok/s). Greedy-device, greedy-cpu, and nongreedy-cpu are
   all in a similar regime — only non-greedy on device is broken.
2. **Going from batch=1 to batch=2 loses only ~5% per-user throughput.** The
   bottleneck is per-decode dispatch / op cost, not batch scaling. So
   batching won't paper over the nongreedy-device cliff.

CPU sampling avoids the device sampler graph entirely, which is why it
dodges the cliff — that's exactly what the device-side hotspots below
quantify.

## Per-step wall-clock (sampler-only, `test_sampler_throughput.py`)

| Config | wall-clock per step | tok/s (sampler-only) |
|---|---|---|
| `top_k=None, top_p=None` (default) | ~23 ms | ~43 |
| `top_k=3` (opt-in)                 | 70 ms  | 14.3 |

Wall-clock comes from the throughput script's host-side `t1 - t0` around
`out.cpu()` (forces device sync). Tracy CSV device-firmware sums are higher
(~87 ms for top_k=3) because per-op firmware durations overlap with
async dispatch — sum is meaningful for **relative** op comparisons, not
end-to-end.

Enabling `top_k=3` adds ~47 ms wall-clock per step → 3× slowdown.

## Hotspots — default path (`top_k=None, top_p=None`)

CSV: `tracy_sampler_nongreedy_baseline/reports/2026_04_27_15_53_06/ops_perf_results_*.csv`
Filter: `--signpost "iter 3"`

```
                                       sum_ms  count  mean_us    pct
OP CODE
FillPadDeviceOperation                  9.113      1 9113.003  39.64
BinaryNgDeviceOperation                 5.403      9  600.364  23.51
SoftmaxDeviceOperation                  5.031      2 2515.742  21.89
ReduceDeviceOperation                   1.854      1 1854.093   8.07
UnaryNgDeviceOperation                  0.443      3  147.658   1.93
TypecastDeviceOperation                 0.385      7   55.004   1.68
TilizeWithValPaddingDeviceOperation     0.206      5   41.166   0.90
TernaryDeviceOperation                  0.185      3   61.604   0.80
ArgMaxDeviceOperation                   0.167      2   83.434   0.73
UntilizeWithUnpaddingDeviceOperation    0.136      2   68.193   0.59
RandDeviceOperation                     0.064      1   63.544   0.28

Total: 22.987 ms across 36 ops (11 distinct op types)
```

**No `SortDeviceOperation`** — `apply_top_k_top_p` returns early when both
`top_k` and `top_p` are `None`, so `probs.sort()` never runs. This is the
typical default tt-cloud-console request shape (per chat-completion logs).

Three ops dominate: `FillPad` (40%), `BinaryNg` (24%), `Softmax` (22%).

## Hotspots — opt-in path (`top_k=3`)

CSV: `tracy_sampler_nongreedy_baseline_topk/reports/2026_04_27_16_38_12/ops_perf_results_*.csv`
Filter: `--signpost "iter 2"`

```
                                       sum_ms  count   mean_us    pct
OP CODE
SortDeviceOperation                    30.137      1 30137.220  34.50
FillPadDeviceOperation                 28.091      3  9363.827  32.16
PadDeviceOperation                      9.164      1  9163.596  10.49
SoftmaxDeviceOperation                  7.545      3  2514.914   8.64
BinaryNgDeviceOperation                 5.362     14   382.965   6.14
TypecastDeviceOperation                 3.192     14   227.976   3.65
ReduceDeviceOperation                   1.752      1  1752.095   2.01
UntilizeWithUnpaddingDeviceOperation    0.487      4   121.784   0.56
UnaryNgDeviceOperation                  0.445      4   111.251   0.51
TernaryDeviceOperation                  0.349      5    69.770   0.40
TilizeWithValPaddingDeviceOperation     0.215      7    30.764   0.25
SliceDeviceOperation                    0.187      2    93.722   0.22
ArgMaxDeviceOperation                   0.173      2    86.496   0.20
ReshapeViewDeviceOperation              0.150      1   149.976   0.17
RandDeviceOperation                     0.110      1   110.186   0.13
EmbeddingsDeviceOperation               0.002      1     1.605   0.00

Total: 87.361 ms across 64 ops (16 distinct op types)
```

**`SortDeviceOperation` = 30.14 ms (34.5%)** — single call, single core. This
is the well-known cliff that commit `87e9a927f` ("Replace sort-based sampling
with multi-core topk for 2.17× speedup", 2026-04-25) is designed to fix.
That commit is **not yet on this branch** (`git branch --contains` returns
empty). Its dependencies (`tt-xla#3729`, `tt-mlir#7504`) need to be verified
against the current submodule pin before cherry-pick.

`FillPad` jumps from 1 call to 3 calls (9.11 → 28.09 ms) and `PadDeviceOperation`
appears for the first time (9.16 ms, 1 call) — both are vocab-size pad work
inside the sort path. `Softmax` rises from 2 to 3 calls. `BinaryNg` and
`Typecast` get more callsites.

## Marginal cost of opting in to `top_k=3`

|  | top_k=None | top_k=3 | Δ |
|---|---|---|---|
| Per-iter device sum   | 22.99 ms | 87.36 ms | **+64.37 ms** |
| Of which: Sort        |  0.00 ms | 30.14 ms | +30.14 ms (47% of Δ) |
| Of which: FillPad     |  9.11 ms | 28.09 ms | +18.98 ms |
| Of which: Pad         |  0.00 ms |  9.16 ms |  +9.16 ms |
| Of which: Softmax     |  5.03 ms |  7.55 ms |  +2.52 ms |
| Of which: BinaryNg    |  5.40 ms |  5.36 ms |  -0.04 ms |
| Of which: other       |  3.45 ms |  7.06 ms |  +3.61 ms |

Sort alone is ~47% of the marginal cost added by the top_k path. Pad-related
work (FillPad delta + new Pad op) is another ~44%. The replacement strategy
in commit `87e9a927f` (split vocab into power-of-2 chunks, run multi-core
ttnn.topk per chunk, candidate-set top-k/top-p) eliminates the Sort and
also reduces the pad work — most of the marginal cost should disappear in
one cherry-pick.

## Targets, in priority order

1. **Sort → multi-core topk.** Cherry-pick `87e9a927f` once submodule pin
   includes its deps. Expected: 30 ms → ~0.7 ms (commit reports 4.43 → 9.60
   tok/s = 2.17× on Llama-3.2-3B, top_p=0.9). Verify with
   `analyze_tracy.py … --vs <after.csv>` against this baseline.
2. **FillPad / Pad in the default path.** 9.1 ms / 40% of no-top-k time
   from a single call. Investigate the pad shape via
   `python perf_debug/analyze_tracy.py CSV --detail FillPad` and check
   whether the padded vocab dimension can be shrunk or the op elided.
3. **Softmax** — 5 ms in default (2 calls × 2.5 ms), 7.5 ms in top_k=3
   (3 calls × 2.5 ms). Each call is steady at ~2.5 ms; reducing call count
   (post-temperature softmax + post-filter softmax could potentially fuse)
   is the lever.
4. **BinaryNg** — 23% in default (9 calls × 0.6 ms). Lots of small elementwise
   ops. Check whether any are fusion-blocked using
   `TTXLA_LOGGER_LEVEL=DEBUG` to dump TTNN IR and grep for adjacent
   `ttnn.add/mul/sub` clusters.

## How to reproduce

```bash
# Default path (no top_k): ~23 ms / step, ~43 tok/s
python perf_debug/test_sampler_throughput.py 2>&1 | tee bench_default.log

# Opt-in top_k path: ~70 ms / step, ~14 tok/s, hits Sort
python perf_debug/test_sampler_throughput.py --top-k 3 2>&1 | tee bench_topk3.log

# Tracy capture (default path)
TRACY_PROFILING_ACTIVE=1 python -m tracy -p -r -o tracy_default \
    -m perf_debug.test_sampler_throughput |& tee tracy_default.log

# Tracy capture (top_k=3)
TRACY_PROFILING_ACTIVE=1 python -m tracy -p -r -o tracy_topk3 \
    -m perf_debug.test_sampler_throughput -- --top-k 3 |& tee tracy_topk3.log

# Default mode: per-iter wall-clock for last 5 iters (catch outliers) +
# op-level hotspot for the last iter only.
python perf_debug/analyze_tracy.py tracy_topk3/reports/*/ops_perf_results_*.csv

# Op-level diff vs this baseline
python perf_debug/analyze_tracy.py tracy_default/reports/*/ops_perf_results_*.csv \
    --vs tracy_after_change/reports/*/ops_perf_results_*.csv
```
