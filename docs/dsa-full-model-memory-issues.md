# DeepSeek-V3.2 full-model (61-layer) vLLM E2E test — memory investigation

**Test:** `tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py::test_tensor_parallel_generation_deepseek_v32_full`
**Hardware:** Blackhole Galaxy, 32 devices, `mesh_shape=[8, 4]`
**Goal:** run all 61 layers of DeepSeek-V3.2 (not the existing 3-layer stub) and generate 10 tokens end to end.

Four runs have failed, each for a different reason. This doc records what's confirmed, what's
hypothesis, and the options for getting past each blocker. Failures #1-#3 are the original
sequence (the then-current blocker being a host-memory stall); #4 was a later *device*-DRAM
regression, since resolved.

## Summary table

| # | Failure mode | Where | Root cause | Status |
|---|---|---|---|---|
| 1 | `TimeoutError` at 3600s | pytest watchdog (`conftest.py::_test_timeout`) | New test had no `.test_durations` entry, so it fell back to the 1-hour default; weight load alone takes ~27 min, compile takes longer | **Fixed** — seeded `.test_durations` with 21600s (mirrors the `DeepSeek-V4-Flash` full-e2e precedent) |
| 2 | `TT_FATAL: Out of Memory` | tt-metal `bank_manager.cpp`, during first graph execution | 256 routed experts (~654B params) sharded 32-way across the mesh still need ~38 GiB/device at bf16, over the ~31.88 GiB per-chip DRAM budget | **Attempted fix**: added `experimental_weight_dtype: "bfp_bf8"` |
| 3 | No crash, no progress — **process stall / host swap-thrashing** | Same point: right after `Compiling graph for config={'num_tokens': 128, ...}` | Host RAM (566 GiB) exhausted; process footprint reached ~1.39 TB (514 GiB resident + 872 GiB swapped), and the host is thrashing swap rather than computing | **Open** — this doc |

| 4 | `TT_FATAL: Out of Memory` (16 MiB buffer, `largest free block: 1294336 B`) | tt-metal `bank_manager.cpp`, in vLLM's `_initialize_kv_caches` | A `TTCore_NonCacheableTrait` added to `MeshPartitionOp` in a local tt-mlir change excluded that op from *device-side* const-eval, moving the const-eval boundary to before the size-reducing shard slice, so cached tensors became the pre-slice (~5x larger) versions | **Fixed** — trait reverted; freed 572 MB of const-eval DRAM at 4 layers, scaling per-layer. Detail: [`dsa_indexer_score_decomposition_rank_fix.md`](./dsa_indexer_score_decomposition_rank_fix.md) 3.5 and 4.4 |

## Failure #3 in detail

### What's directly observed (confirmed)

- The log (`full_e2e_new.log`) went silent at `2026-08-05 20:24:21` — the same point in the pipeline where run #2
  crashed (right after entering `Compiling graph for config={'num_tokens': 128, ...}`). At time of investigation it
  had been **5+ hours** with zero new log lines and an unchanged line count.
- `EngineCore` (the vLLM child process that does all the real work) was still alive: `VmRSS = 514 GB`,
  `VmSwap = 872 GB` — a combined footprint of **~1.39 TB**.
- Host: 566 GiB total RAM (560 GiB used, 2.6 GiB free), swap 1.0 TiB total (833 GiB used, 190 GiB free).
- `vmstat` showed a sustained swap-out rate of ~235 MB/s across every sample, near-zero swap-in, CPU 96-98% idle,
  `wa` (I/O wait) near 0% — the standard signature of **thrashing**: the kernel continuously evicts pages to make
  room, the process makes essentially no forward progress, and it isn't blocked on real device or disk work.
- Of 616 threads, only 1 was ever caught in state `R`; the rest sat in `futex_wait_queue` / `hrtimer_nanosleep`
  (idle threadpool workers) — consistent with almost nothing actually executing.
- This is a **new** failure mode. Runs #1 and #2 both reached this same pipeline stage at essentially the same
  elapsed time (~55-58 min post-start) and either timed out or crashed within another minute or two — neither
  showed multi-hour silence or a multi-hundred-GB process footprint. The only code change between run #2 and run
  #3 was adding `experimental_weight_dtype: "bfp_bf8"`.

### Likely mechanism (hypothesis — not confirmed)

`experimental_weight_dtype` is a PJRT/tt-mlir **compile-time** option (`platform.py:116`, threaded through
`get_pjrt_compile_config()`), not the Python-level `weight_dtype_overrides` mechanism
(`tt_torch/weight_dtype.py`, which inserts a `torch.ops.tt.weight_dtype_override` custom op via
`torch.nn.utils.parametrize` — a different code path entirely). The coarse flag's actual conversion logic lives in
the C++ compiler, which wasn't inspected as part of this investigation, so the exact mechanism is unconfirmed.
Two plausible explanations, either or both of which could be true:

1. **Weight staging.** Converting ~654B params to `bfp_bf8` during compilation may require the compiler (running
   on the host) to hold the original bf16 tensors and the converted bfp8 tensors simultaneously at some point,
   roughly doubling the peak host footprint versus a bf16-only compile.
2. **Compiler/IR memory scaling.** Quantization inserts extra convert/pack ops throughout a graph that's already
   large (61 layers × 32-way Shardy partitioning). MLIR-style compilers can scale super-linearly in memory with IR
   size and pass complexity, especially at `optimization_level=0` (all optimizations disabled, per `platform.py`'s
   own docstring) — the host-side compiler process itself, not weight staging, could be the memory sink.

Neither is confirmed. It's also possible plain bf16 (no quantization) would show the *same* stall if the model
were pushed slightly further (e.g. one more layer, or a larger prefill bucket) — runs #1/#2 didn't linger long
enough at this stage to say for sure whether they were also trending toward a large host footprint before they
failed for other reasons first.

### One relevant, already-documented data point

`build_weight_cache.py`'s own docstring: *"Processes each layer independently so peak memory is ~23 GB (one MoE
layer) instead of ~1.37 TB (all 61 layers)."* That "~1.37 TB" figure — for loading all 61 layers of this model
**unchunked** — lines up closely with the ~1.39 TB observed here. That comment is about the offline cache-builder
script, not the vLLM runtime loader, but it strongly suggests vLLM's stock loader (`default_loader.py`, seen in
the log) is materializing the full checkpoint in host memory rather than loading it layer-by-layer the way the
cache builder deliberately avoids doing.

## Potential ways to get around it

Roughly in order of how cheaply they isolate the cause vs. how much engineering they require:

1. **Isolate the trigger first.** Re-run at a much smaller layer count (e.g. 16-24 layers — still hits MoE, still
   exercises `bfp_bf8`) with the same `[8,4]` mesh and watch host RSS/swap over time. If host memory blows up
   disproportionately to model size even at small scale, that points at compiler/IR overhead rather than weight
   staging (or vice versa). Cheap, fast, and answers the open question above before spending more engineering time.

2. **Bake the reduced-precision weights offline**, extending `build_weight_cache.py`'s existing per-layer chunked
   approach (already used to keep the bf16 dequant peak at ~23 GB/layer) to also emit a pre-packed low-precision
   checkpoint on disk. vLLM would then load an already-small file directly instead of converting a full bf16
   checkpoint at compile time. This avoids whichever of the two hypotheses above is real, at the cost of writing
   new block-fp8 packing logic and confirming vLLM's loader/quant machinery accepts a pre-packed checkpoint
   directly.

3. **Check whether the loader can stream/chunk instead of materializing the whole model.** Look at
   `default_loader.py` (vLLM stock loader) and `vllm_tt/model_runner.py`'s weight-loading path for a
   shard-and-free-per-layer option, mirroring what `build_weight_cache.py` already does offline. This is the
   "real" fix if hypothesis 1 (weight staging) is correct, but likely the largest lift of the options here.

4. **Try `weight_dtype_overrides` / `apply_weight_dtype_overrides()` instead of the coarse `experimental_weight_dtype`
   flag.** It's a different code path (a graph-level custom op inserted in Python before compile via
   `torch.nn.utils.parametrize`, not a compiler-internal conversion pass), so it may have different — untested —
   host-memory behavior. `docs/source/mixed_precision.md` notes it's currently validated for matmul/linear layers,
   not confirmed for `TTFusedMoE` experts, so this needs verification either way.

5. **Infra-level mitigations**: more host RAM, faster swap storage (if current swap is on slow storage, thrashing
   is proportionally worse), or running on a host with more memory if one is available. Blunt, no code changes,
   but doesn't address the underlying inefficiency.

6. **Kill the current stuck run.** At ~235 MB/s sustained swap-out with zero log progress in 5+ hours, it is very
   unlikely to complete in any practical amount of time, and it's holding ~1.39 TB of host memory that could be
   needed elsewhere on a shared machine.

## Where things stood at time of writing

- `EngineCore` PID 48915 (child of pytest PID 48710, running in a `tmux` session) was still alive and thrashing,
  last log line at `2026-08-05 20:24:21`, `full_e2e_new.log` at 9506 lines.
- No fix for failure #3 has been applied yet to the test itself — `experimental_weight_dtype: "bfp_bf8"` is still
  in `test_tensor_parallel_generation_deepseek_v32_full`'s `additional_config` pending the investigation above.

## Failure #4 (later): device-DRAM OOM from const-eval cache inflation — RESOLVED

Distinct from #2/#3 and worth separating: not weights, not host RAM, and nothing to do with quantization.
This one was **self-inflicted by a local tt-mlir change** and is fully resolved.

Symptom — the run compiled for ~8.6 h, then died at the last step of engine startup:

```
TT_FATAL: Out of Memory: Not enough space to allocate 16777216 B DRAM buffer across 8 banks,
where each bank needs to store 2097152 B, but bank size is 4272341376 B
(allocated: 4269564032 B, free: 2777344 B, largest free block: 1294336 B)
```

Two details make this diagnosable:

* It is **fragmentation**, not a clean capacity wall — 2.6 MiB free per bank, but the largest
  contiguous block is 1.23 MiB against a 2 MiB-per-bank request, at ~99.9% occupancy.
* The failing buffer is a **const-eval cached tensor** (`ttcore.load_cached` → `ttnn.to_device`
  in the backtrace), specifically graph `g0`'s `tensor<2048x2048xsi32>` = exactly 16777216 B.

Root cause: a `TTCore_NonCacheableTrait` added to `TTIR_/TTNN_MeshPartitionOp`. That trait's only
consumer (`ConstEvalHoist.cpp:205`) also excludes the op from **device-side** const-eval, where
hoisting is safe. Because `mesh_partition` is the *size-reducing* op (it slices a global tensor down
to one device's shard), excluding it pushed the const-eval boundary to *before* the slice — so the
persistent cached tensors became the pre-slice, `num_devices`x larger versions. The trait was also
redundant: upstream `CPUHoistConstEval.cpp:88-92` already keeps `mesh_partition` off the CPU.

Measured with `test_dsa_v32_4layer_ccl_ir_diagnostic` (4 layers, same mesh/opt/quant config as the
full test, ~10 min instead of ~9 h):

| | trait ON | trait reverted |
|---|---|---|
| `mesh_partition` inside const-eval funcs | 0 of 75 | 69 of 75 |
| const-eval function count | 479 | 270 |
| **total const-eval cached bytes** | **2255.0 MB** | **1683.0 MB** |

**-572 MB of persistent DRAM at four layers**, scaling per-layer — far more headroom than the ~14 MB
the failing allocation was short by. Generated tokens were bit-identical before and after, so this was
purely a memory regression with no numerics impact.

Two takeaways for future memory work here:

* The DRAM budget is tight enough (~99.9% occupancy at 61 layers) that a few hundred MB of extra
  *persistent const-eval caching* is the difference between running and not. Where const-eval
  boundaries fall relative to sharding/slicing ops is therefore a first-class memory concern, not
  just a compile-time detail.
* `test_dsa_v32_4layer_ccl_ir_diagnostic` exists specifically to make that measurable cheaply —
  it prints CCL counts and copies the exported IR to `/tmp/dsa_4l_ccl_ir` for before/after diffing.
