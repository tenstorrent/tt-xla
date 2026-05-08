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

## Severity assessment for our autoresearch loop

Neither bug blocks the loop. Bug 1 cost us one ~26-min round trip to discover (we then switched to `dry_run=True`). Bug 2 cost ~84 s for the failed standalone run plus ~1 min to identify and patch. Both workarounds are stable.

For a tt-xla user running `codegen_py` for the first time on any sharded multi-chip model, both are likely to surface — Bug 2 in particular hits the very first attempt to run the emitted artifact. Worth fixing both before the codegen path is recommended for production use.

---

## How these were found

Discovered while building `dgolubovic/autosearch-mp-gpt-oss-120b` — an autoresearch loop ([uditgoenka/autoresearch](https://github.com/uditgoenka/autoresearch)) tuning per-op compute knobs in the emitted Python TTNN of GPT-OSS 120B for the same Galaxy 4×8 system, mirroring the [svuckovic/dit-autotune](https://github.com/tenstorrent/tt-metal/compare/main...svuckovic/dit-autotune) campaign from 2026-04-30.

Full session report: `autoresearch_logs/PROGRESS_REPORT.md` in this branch.
