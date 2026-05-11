# Codegen bugs found while wiring up the GPT-OSS 120B autoresearch loop

Reproducible on `dgolubovic/autosearch-mp-gpt-oss-120b` of `tenstorrent/tt-xla` against tt-mlir SHA `f8d3bf0e97dee04ea1783b00304b37b48d446c62` (uplift PR #4450, 2026-05-04). Both bugs are in the `codegen_py` emit path. None block the autoresearch loop, but they make the emitted artifact harder to use than it should be.

---

## Bug 1 — `dry_run=False` execution shadowed by test framework's `utils.py`

**Symptom:**

```
RuntimeError: Traceback (most recent call last):
  File ".../graph_0/main.py", line 4809, in main_for_test
  File ".../graph_0/main.py", line 4447, in _main
  File ".../graph_0/main.py", line 123, in capture_or_execute_trace_0_main
  File ".../graph_0/main.py", line 4269, in run_and_capture_trace_0_main
AttributeError: module 'utils' has no attribute 'DeviceGetter'
```

**Repro:**

```bash
# In the tt-xla repo
docker exec --user $(id -u):$(id -g) --workdir /home/dgolubovic/repos/tt-xla tt-xla-ird-dgolubovic bash -lc '
  source venv/activate && cd tests/benchmark && \
  CODEGEN_EXPORT_PATH=/tmp/codegen_repro python -m pytest -svv \
    test_llms.py::test_gpt_oss_120b_tp_galaxy_batch_size_64 --accuracy-testing --num-layers 1
'
```

with `tests/benchmark/benchmarks/llm_benchmark.py` setting `options["backend"] = "codegen_py"` + `options["dry_run"] = False` (see commit `8867a0489` in this branch).

**Root cause:**

The `codegen_py` backend with `dry_run=False` re-executes the emitted `main.py` after writing it to disk. This re-execution happens from inside pytest's process, where `sys.path` already contains `tests/benchmark/` (the test framework root). The emitted `main.py` does:

```python
import ttnn
import utils
```

`import utils` resolves to `tests/benchmark/utils.py` (which has `build_xla_export_name`, `compute_pcc`, `print_benchmark_results`, etc.) instead of the codegen-emitted `<export_path>/graph_N/utils.py` (which has `DeviceGetter`, `load_tensor`, etc.). The framework's `utils` shadows the emitted one. The first call into emitted code that hits `utils.DeviceGetter.get_device(...)` throws `AttributeError`.

Confirming evidence:
- `tests/benchmark/utils.py` is on `sys.path` because pytest's CWD is `tests/benchmark/`.
- Emitted `<export_path>/graph_N/utils.py` does define `DeviceGetter` correctly (verified in our run).
- Emitted `<export_path>/graph_N/main.py` does `import utils` (line 2), not `from . import utils`.
- Switching to `dry_run=True` (commit `9d0e086c5` in this branch) avoids the re-execution and the bug.

**Suggested fix (pick one):**

1. **Emit relative imports.** Change codegen's `main.py` template to emit `from . import utils` (and add a sibling `__init__.py`, which already exists). This makes the emitted package self-contained regardless of how it's invoked.
2. **Rename the emitted module.** Emit `_codegen_utils.py` (or similarly unique) and `import _codegen_utils as utils`. Avoids any chance of shadowing by host-side `utils` modules.
3. **Insert `<export_path>/graph_N` to the front of `sys.path` before re-executing in dry_run=False.** Ensures the emitted dir's `utils.py` wins over the framework's.

(1) is cleanest and idiomatic Python. (2) is a one-character change in the codegen template. (3) is a runtime-only fix and feels brittle.

**Workaround used in this loop:** flipped to `dry_run=True` so the emit isn't followed by an in-process re-execute. Costs us the "verify codegen output matches flatbuffer" check that `dry_run=False` was meant to provide (per `examples/jax/codegen/python/emitpy_execute.py:42-50`).

---

## Bug 2 — Emitted `utils.DeviceGetter` sets `FABRIC_1D` but emitted `main.py` uses `Topology.Ring`

**Symptom:**

When `python3 graph_N/main.py` is run standalone (or via the emitted `run` script):

```
RuntimeError: TT_FATAL @ tt_metal/fabric/fabric.cpp:153: forwarding_direction.has_value()
info: Could not find any forwarding direction from src (M0, D0) to dst (M0, D3)
backtrace:
 --- ttnn::build_ring_reduce_scatter_minimal_async_program_artifacts(...)
 --- ttnn::operations::ccl::ReduceScatterDeviceOperation::ReduceScatterProgram::create_at(...)
 --- ttnn::reduce_scatter(...)
```

**Repro:**

```bash
# After a successful codegen emit (with dry_run=True), then:
docker exec --user $(id -u):$(id -g) --workdir /home/dgolubovic/repos/tt-xla tt-xla-ird-dgolubovic bash -lc '
  source venv/activate && \
  cd /home/dgolubovic/repos/tt-xla/autoresearch_logs/codegen_artifacts/gpt_oss_120b_1lyr/graph_0 && \
  bash run
'
```

**Root cause:**

The emitted `graph_N/utils.py` has, in `DeviceGetter.get_device(mesh_shape)`:

```python
if math.prod(mesh_shape) >= 2:
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
```

— linear/non-ring fabric. The emitted `graph_N/main.py` calls `reduce_scatter` with explicit ring topology in many places:

```python
ttnn_reduce_scatter_0 = ttnn.reduce_scatter(
    ...,
    topology=ttnn.Topology.Ring,
)
```

— grep confirms multiple `topology=ttnn.Topology.Ring` callsites in `graph_0/main.py` (1322, 1430, 1456, 1689, 1715, 2567, …). Ring topology requires `FabricConfig.FABRIC_1D_RING` (or comparable) for the fabric layer to find a forwarding direction; with `FABRIC_1D` (linear) it cannot route from `M0:D0` to `M0:D3` (i.e., wrap-around) and throws.

Available `ttnn.FabricConfig` values: `CUSTOM, DISABLED, FABRIC_1D, FABRIC_1D_NEIGHBOR_EXCHANGE, FABRIC_1D_RING, FABRIC_2D, FABRIC_2D_TORUS_X, FABRIC_2D_TORUS_XY, FABRIC_2D_TORUS_Y`. The natural match for ring-topology reduce_scatter is `FABRIC_1D_RING`.

**Suggested fix (pick one):**

1. **Emit fabric config from observed graph topology.** During codegen, scan the emitted ops for any `topology=Topology.Ring` / `Topology.Linear` / etc., and emit `DeviceGetter.set_fabric_config(...)` with the matching fabric. If both exist, prefer the more general one (e.g., `FABRIC_1D_RING` covers linear too).
2. **Move fabric config out of `DeviceGetter` into `main.py`.** Have `main()` set the fabric explicitly before `DeviceGetter.get_device(...)`, with the value baked in by codegen based on the graph it's emitting. Keeps `DeviceGetter` a generic helper.
3. **Auto-detect fabric needs at runtime.** `DeviceGetter` could open with no fabric initially, then re-set fabric the first time a ring op is encountered. Probably too clever.

(1) is the most direct fix.

**Workaround used in this loop:** patched the emitted `graph_*/utils.py` files with `sed` to swap `FABRIC_1D` → `FABRIC_1D_RING`. Confirmed standalone `bash graph_0/run` then completes cleanly (151s end-to-end on 1-layer GPT-OSS 120B).

---

## Bug 3 — Codegen-emitted full GPT-OSS 120B OOMs at runtime; root cause not the dtype-override path (open)

**Symptom:**

Codegen of a full-size GPT-OSS 120B (`pytest test_gpt_oss_120b_tp_galaxy_batch_size_64 --accuracy-testing` with `CODEGEN_EXPORT_PATH=…` and `weight_dtype_overrides={…experts.{gate_up,down}_proj: bfp_bf4, router: bfp_bf4}`) succeeds and writes ~872 GB across 4 graphs (218 GB / graph, 11h 48m wall on tt-mlir `f8d3bf0e`). When the emitted code is executed (via the standalone `bash run` script or the in-process harness), `_main` fails with:

```
TT_FATAL: Out of Memory: Not enough space to allocate 377487360 B DRAM buffer
across 12 banks, where each bank needs to store 31457280 B, but bank size is
1071821792 B (allocated: 1017920704 B, free: 53901088 B,
largest free block: 31453440 B)
```

The same test in non-codegen mode on the same Galaxy reservation runs successfully at TOP1=81.25% / TOP5=95.31% (37 min wall, decode 568 ms). So the model fits on this hardware with the configured dtype overrides applied; only the emitted-code execution path doesn't fit.

**What is NOT the bug (despite my initial guess):**

I first suggested the `weight_dtype_overrides` (`apply_weight_dtype_overrides` torch parametrization) was being dropped from the emit, citing a `grep` for `to_dtype` returning zero in the emitted `main.py`. That was the **wrong needle** — the conversion lowers to `ttnn.typecast`, not `to_dtype`, in this pipeline.

The right checks come back clean:

```bash
# (1) ttnn.typecast IS emitted into main.py — 403 calls, with the expected target dtypes
$ grep -c 'ttnn\.typecast(' <export_path>/graph_2/main.py
403
$ grep -A1 'ttnn\.typecast(' <export_path>/graph_2/main.py | grep -oE 'DataType\.\w+' | sort | uniq -c
    146 DataType.FLOAT32     # intermediates / accumulators
    108 DataType.BFLOAT4_B    # = bfp_bf4 — matches the experts.{gate_up,down}_proj + router count
     73 DataType.BFLOAT8_B    # = bfp_bf8 — matches conftest's default experimental_weight_dtype
     38 DataType.BFLOAT16     # activations into matmul inputs
     37 DataType.INT32
      1 DataType.UINT32

# Pattern is from_device → typecast(target) → deallocate(host bf16) → to_device(DRAM)
# — exactly the chain HOST_BFP_PACKING_VALIDATION_NOTES.md describes.

# (2) ttcore.weight_dtype arg attributes survive end-to-end through the MLIR pipeline:
#   shlo (raw):                  108 weight_dtype, 0 typecast
#   shlo_frontend/compiler:        1 (global)
#   ttir:                          1 weight_dtype, 222 typecast
#   ttnn:                        109 weight_dtype, 403 typecast
#   emitted main.py:               —, 403 typecast
```

The 108 SHLO-level arg annotations match the 108 weights targeted for bfp_bf4 by the test's `weight_dtype_overrides` glob. They get folded into the global option through the frontend, then expanded back into per-weight typecast ops at TTIR (222 — two per overridden tensor) and TTNN (403 — including additional intermediate conversions). **The dtype-override path is healthy.**

That `.tensorbin` files are full bf16 on disk is *expected* with this design — the chain executes at runtime (or via a const-eval cache; see notes below) to materialize the bfp representation, the tensorbins just hold the source data.

**What IS likely the bug (still undiagnosed):**

The allocator log shows the failure isn't simple total-memory exhaustion — it's *fragmentation*:

```
per-bank size:        1,071,821,792 B  (~1.07 GB)
per-bank allocated:   1,017,920,704 B  (~95 % full)
per-bank free:           53,901,088 B  (~54 MB)
largest free block:      31,453,440 B  (~31 MB)
required per bank:       31,457,280 B  (~31 MB)
```

Each bank has enough free space in aggregate, but the largest contiguous chunk is ~3 KB shy of the request. That points at allocation *order* / *lifetime* differing between the production path and the codegen-emit path, not at the dtype path.

Plausible avenues (would each need a focused look — I haven't bisected):
- **Intermediate tensors not freed when production would free them.** Emit captures one specific allocation order from the graph trace; if a `ttnn.deallocate(...)` call missed an intermediate, that buffer stays resident through subsequent matmuls.
- **Const-eval cache materialization differs.** The user's note: "even with the typecast chain emitted, whether the const-eval result is materialized to disk as bfp depends on how codegen serializes const-eval outputs." If the const-eval ran at codegen time (or first-run with caching), the runtime allocator sees a different working-set than production.
- **Sharding / mesh-axis differences** that change per-device tensor sizes by enough to push past the per-bank threshold. The user's edited test uses the loader's default `get_mesh_config` / `load_shard_spec` rather than explicit `mesh_config_fn`; codegen-time vs runtime resolution of those defaults could differ.

**Suggested next diagnostic (per user):**

> Fastest reproducer would be running `ttmlir-opt --ttir-to-emitpy-pipeline` on the captured MLIR and dumping IR between passes (`--mlir-print-ir-after-all`) to see where the allocation chain disappears / diverges from the production lowering.

Specifically: compare the TTNN-level MLIR (with explicit memory_config attributes on each op) for the codegen run vs. an equivalent production run, and see which ops have different per-tensor `BufferType` / `TensorMemoryLayout` choices.

**Severity:** showstopper for full-120B autoresearch on this branch until diagnosed. Workaround used in this loop: fall back to the 1-layer artifact (33 GB) for loop *mechanism* validation only. Real per-op MP knob research on full GPT-OSS 120B waits on a fix.

**Cost so far:** ~12 h Galaxy time on the codegen + ~3 min on the failed harness execution. 872 GB artifact retained for debugging; can be deleted to reclaim disk once the diagnosis lands upstream.

---

## Severity assessment for our autoresearch loop

- **Bug 1** cost ~26 min (one full codegen round trip) to discover; switching to `dry_run=True` is a stable workaround.
- **Bug 2** cost ~84 s for the failed standalone run plus ~1 min to identify and patch (`sed` `FABRIC_1D` → `FABRIC_1D_RING`); stable workaround.
- **Bug 3 is a showstopper** for the loop on full GPT-OSS 120B. Burned ~12 h of Galaxy hardware time on the codegen, then ~3 min on the failed harness execution, with no usable artifact at the end. Until the dtype-override path is fixed, real-model autoresearch on this branch can't proceed beyond the 1-layer mechanism validation.

For a tt-xla user running `codegen_py` for the first time on any sharded multi-chip model, all three are likely to surface in this order — Bug 2 hits the first attempt to run; Bug 1 hits the first attempt to use `dry_run=False`; Bug 3 hits the first attempt to use `weight_dtype_overrides` at any non-trivial scale. Worth fixing all three before the codegen path is recommended for production use.

---

## How these were found

Discovered while building `dgolubovic/autosearch-mp-gpt-oss-120b` — an autoresearch loop ([uditgoenka/autoresearch](https://github.com/uditgoenka/autoresearch)) tuning per-op compute knobs in the emitted Python TTNN of GPT-OSS 120B for the same Galaxy 4×8 system, mirroring the [svuckovic/dit-autotune](https://github.com/tenstorrent/tt-metal/compare/main...svuckovic/dit-autotune) campaign from 2026-04-30.

Full session report: `autoresearch_logs/PROGRESS_REPORT.md` in this branch.
