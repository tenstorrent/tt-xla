# Low-batch streaming MoE (gpt-oss-20b) — handoff

Streams only the router-selected top-k experts from DRAM for **batch=1 decode**,
instead of computing all 32 experts dense. Runs in the compiled tt-xla/PJRT flow via
an opaque `tt_lang_op` that lowers to a `ttnn.generic_op` with a custom JIT kernel.
End-to-end on gpt-oss-20b (8-device WH TP), trace=on, and **beats the dense baseline**
at batch=1.

## TL;DR
- **Branch (all 3 repos): `sshon/wip-1batch-moe`.** One fresh `tt-xla` clone @ this
  branch builds the whole stack via the version pins (no submodules; CMake clones
  tt-mlir @ pinned SHA, which clones tt-metal @ pinned SHA).
- Default weight dtype = **bfp4** (`_IN1_DTYPE` in `stream_experts_kernel.py`).
- **Streaming wins at batch=1, loses at high batch** — it re-streams weights per
  (token,expert) row, so it's a *low-batch-only* optimization. At batch=16 dense wins.

## Repro branch SHAs (pin chain)
| repo | commit | pins |
|------|--------|------|
| tt-xla  | `1008d8e0` | `TT_MLIR_VERSION = 97cfe193bb7f4dfd4599a4da3318c0f6eb46b05f` |
| tt-mlir | `97cfe193` | `TT_METAL_VERSION = d97c2c10b8f9aced6a758143f25cf01fe5cd4b2f` |
| tt-metal| `d97c2c10` | (interleaved experts kernel) |

`TT_MLIR_VERSION` lives in `tt-xla/third_party/CMakeLists.txt`; `TT_METAL_VERSION` in
`tt-mlir/third_party/CMakeLists.txt`. The local "reuse prebuilt tt-metal" hack
(configure/build/install skipped) was **reverted on the branch** so a fresh clone
actually builds tt-metal.

## Build & run (new host)
```bash
git clone -b sshon/wip-1batch-moe git@github.com:tenstorrent/tt-xla.git
cd tt-xla && source venv/activate          # set TTMLIR_TOOLCHAIN_DIR, TTXLA_ENV_ACTIVATED
cmake -G Ninja -B build && cmake --build build   # clones+builds tt-mlir -> tt-metal (slow, hrs)

# full model, trace on, no profiling
pytest -svv tests/benchmark/test_llms.py::test_gpt_oss_20b_tp_batch_size_1_stream --decode-only
# 2-layer quick check
pytest -svv tests/benchmark/test_llms.py::test_gpt_oss_20b_tp_batch_size_1_stream --num-layers 2 --decode-only
# dense baseline for comparison
pytest -svv tests/benchmark/test_llms.py::test_gpt_oss_20b_tp_batch_size_1 --num-layers 2 --decode-only
```
After editing the kernel (`interleaved_experts_matmul_kernel.cpp`) or python you do
NOT need to rebuild — kernels are JIT'd and python is live. Rebuild only for the
tt-mlir passes / emitter / pjrt C++.

## Baseline vs streaming (decode samples/s, trace=on)
| config | 2 layers | 24 layers (full) |
|--------|----------|------------------|
| **dense baseline** | 83.296 | 11.435 |
| stream bf16 | 83.311 (PCC .9994) | — |
| stream bfp8 | 85.837 (.9994) | **11.790** (+3.1%, PCC .999) |
| stream bfp4 | 86.054 (.9947) | (run in progress) |
| **stream bfp4 + cap=12 + in0-reuse** | **88.984 (+6.8%)** | — |

The end-to-end edge is only ~3–7% because (a) attention/LM-head dominate decode and
the MoE is a small slice, and (b) dense ALSO uses bfp8 experts so the win is purely
reading 4/32 experts. The per-op MoE speedup is much larger (see profiling).

## What was tried → result
1. **WIDTH_SHARDED b1 kernel through compiled flow** — IMPOSSIBLE: the opaque
   `tt_lang_op` hands DRAM-interleaved operands; no operand-layout control. → pivoted to Path A.
2. **Path A: new interleaved-DRAM multi-core kernel** (every operand plain interleaved
   DRAM via `InterleavedAddrGenFast`, N columns split across cores). single-core 72.5
   → multi-core ~dense parity.
3. **bfp weights (bfp8/bfp4)** via `weight_dtype_override` on the opaque op (needed 3
   tt-mlir passes; annotation-only override was being skipped) → past dense (86.054).
4. **core cap sweep**: cap=6 (86.3) < cap=12 (87.3) < cap=24 (88.2); **cap=12 best**.
5. **in0 reuse** (read the shared activation once per token, not per output row) →
   **88.984 (+6.8%)**. This was the biggest single win — in0 redundancy was the bottleneck.
6. **b1 read pipelining (trid multi-buffer)** — measured ~neutral/worse, reverted
   (interleaved per-tile reads already overlap).
7. **WIDTH_SHARDED in1 (bank-parallel weight read)** — works single-device (PCC .993)
   but **breaks in the 8-device mesh (PCC .886)** (bank-placement mismatch) AND perf
   gain was within device noise. **Reverted.** (The reshard mechanism via
   `TtLangWidthShardPattern` + a dual-mode kernel is documented in git history if revisited.)

## Tracy profiling (per-op device breakdown)
**Gotcha: needs `pip install pandas==2.3.3`** (repo pins 3.0.0; pandas 3.0's Arrow
strings crash `process_device_log.py:143`). The CI perf job pins 2.3.3 for this reason.
```bash
# trace must be OFF for per-op profiling (flip trace_enabled=False in the stream test temporarily)
python -m tracy -p -r --sync-host-device -o OUT -m pytest -svv \
  tests/benchmark/test_llms.py::test_gpt_oss_20b_tp_batch_size_1_stream \
  --num-layers 1 --decode-only --max-output-tokens 3
# named report: OUT/reports/<ts>/ops_perf_results_<ts>.csv  (OP CODE + DEVICE FW DURATION)
# post-processing is slow (file cleanup); WAIT for it, don't kill early.
```
The benchmark emits signposts; the CSV has warmup then measured sections — use the
**second** `decode_1` (post-prefill) for steady-state numbers. Our streaming MoE shows
up as `GenericOpDeviceOperation` (gate_up output X=768, down output X=2880); dense MoE
is `MatmulDeviceOperation` with `Z=4` (the 4 experts/device from expert-parallel).

### MoE cost, per-device, one decode step (1 layer)
| | gate_up | down | sum | whole layer |
|--|--|--|--|--|
| dense bfp8 (matmul Z=4) | 383 us | 247 us | **630 us** | 10.03 ms |
| stream bfp4 (GenericOp) | 69 us | 45 us | **114 us** | 8.42 ms |
| stream bfp8 (GenericOp) | 120 us | 54 us | **173 us** | — |

- **dtype-matched (both bfp8): streaming MoE is ~3.6x cheaper** than dense (173 vs 630 us).
  (The 5.5x you get comparing stream-bfp4 to dense-bfp8 includes the bfp4 dtype edge.)
- **Whole-layer: stream(bfp4) is 1.19x** faster than dense(bfp8) on device FW
  (611 vs 831 ms summed over 8 dev) — MoE is a small share, attention/head are identical.

## Why the sharding differs (dense vs stream)
| experts weight | dense | stream |
|--|--|--|
| gate_up_proj | `("model",None,None)` = **expert-parallel** (4 experts/device, full weights) | `(None,None,"model")` = **TP col-parallel** (all experts, N sliced) |
| down_proj | `("model",None,None)` expert-parallel | `(None,"model",None)` = **TP row-parallel** → all-reduce after down |

Stream uses TP-within-expert so **every device can independently select the global
top-4** (expert-parallel can't — a device only holds 4 experts). The price is a
`down` all-reduce. Stream also pads the intermediate **2880 → 3072** (so 3072/8=384 is
tile-aligned; dense keeps 2880 whole, no pad) — stream computes +6.7% yet still wins.

## Known optimization opportunities
- **all_reduce lowered as all_gather + local sum** (the naive form): tt-mlir's
  `TTNNAllReduceWorkarounds` (`rewriteAsAllGatherLocalReduce`) decomposes the down
  `ttnn.all_reduce` into `ttnn.all_gather` (gather all 8 partials) + `ttnn.sum`. This
  makes stream's all_gather ~3.3x bigger per call than dense's (410 vs 125 us) and
  costs ~0.42 ms/device. A proper **ring all-reduce / reduce_scatter+all_gather** would
  reclaim most of that and push the whole-layer ratio toward ~1.3x. (ttnn has native
  `all_reduce`/`reduce_scatter`; the workaround likely exists for a CCL hang.)
- WIDTH_SHARDED weight read: re-attempt only if the mesh bank-placement is solved.

## Gotchas
- **Push over SSH** (`git@github.com:tenstorrent/...`); https has no creds here.
- All 3 repos have pre-commit (clang-format/black) that reformats then blocks the
  commit — just `git add` again and recommit.
- **Device ETH-heartbeat wedge**: a hung/killed tracy run can leave an ETH core wedged
  (`ETH core heartbeat check failed ... e4-x`); fresh runs then fail at device init in
  ~1s. Recover with `tt-smi -r` (resets the 4 PCIe cards), then verify
  `python -c "import torch_xla.runtime as xr; print(xr.global_runtime_device_count())"`.
- Logs convention: `tt-xla/tmp/log/run{i}_<name>.log`.

## Key files (on the branch)
- **tt-metal**: `models/demos/deepseek_v3_b1/micro_ops/dram_streaming_experts_matmul/kernels/interleaved_experts_matmul_kernel.cpp` — the kernel (NCRISC reader / TRISC compute / BRISC writer; CT args POSITIONAL; in0-reuse via `k_reuse`; per-core N slice).
- **tt-xla** `python_package/tt_torch/`:
  - `stream_experts_kernel.py` — artifact builder (`_IN1_DTYPE`, `_MAX_CORES=12`, core/RT-arg layout).
  - `custom_ops.py` — `stream_experts_matmul` XLA op (reshapes, bfp override, the CSE-out fix).
  - `moe_stream.py` — host expert padding + the TP stream sharding helper.
  - `moe_backend.py`, `__init__.py`, `tt_lang.py` — backend registration / tt-lang bridge.
- **tt-xla** `pjrt_implementation/.../module_builder/tt_lang_bridge.{h,cc}` — embedded-Python bridge emitting the raw `ttnn.generic_op` artifact.
- **tt-mlir**: `TTNNToFlatbuffer.cpp` (raw_generic emitter + common_addr_operands), `RegisterCustomShardingRule.cpp` (the selective-experts Shardy rule), `PropagateWeightDtype.cpp` / `TTIRToTTNN.cpp` / `TTNNWeightDtypeConversion.cpp` (bfp on the opaque op), `runtime/.../common.h` (BFP_BFloat4 → Bfp4_b).
