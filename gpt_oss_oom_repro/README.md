# GPT-OSS 120B codegen OOM — reproduction bundle

Self-contained reproduction bundle for the bug filed as *"gpt oss 120B codegen OOMs"*. Carries the codegen-emitted Python TTNN (minus the 870 GB of weight tensorbins) plus the corresponding production-path MLIR dumps for diff-style diagnosis.

**Full writeup:** [`CODEGEN_BUGS.md`](../CODEGEN_BUGS.md) (Bug 3) at the repo root.
**Branch:** [`dgolubovic/autosearch-mp-gpt-oss-120b`](https://github.com/tenstorrent/tt-xla/tree/dgolubovic/autosearch-mp-gpt-oss-120b).
**Test:** `tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_galaxy_batch_size_64 --accuracy-testing` on Galaxy 4×8.
**tt-mlir pin:** `f8d3bf0e97dee04ea1783b00304b37b48d446c62` (PR #4450, 2026-05-04 uplift).

---

## TL;DR of the bug

Same test on the same hardware:

| Path | Result | Notes |
|---|---|---|
| Regular pytest (no codegen) | ✅ **TOP1=81.25%, TOP5=95.31%**, 37 min wall | the production baseline |
| Codegen emit (`CODEGEN_EXPORT_PATH=…`) | ✅ exit 0, 11h 48m wall, 872 GB artifact | all 4 graphs emitted |
| **Codegen artifact execution** | ❌ **`TT_FATAL: Out of Memory`** in `_main` of `graph_2` | allocator fragmentation, not exhaustion |

Allocator log on the OOM:
```
Required: 31457280 B per bank
Largest free block: 31453440 B  (~3 KB short of fitting)
Per-bank free: 53901088 B  (~54 MB total, fragmented)
Per-bank size: 1071821792 B  (~1.07 GB)
```

Production path and codegen path lower from the same StableHLO root and same `apply_weight_dtype_overrides` (`bfp_bf4` on `experts.{gate_up,down}_proj` + `router`). Memory budget is the same. Allocation order / intermediate-buffer lifetimes diverge between the two lowerings; the codegen path fragments enough to fail.

---

## Layout

```
gpt_oss_oom_repro/
├── README.md                                 # this file
├── codegen_artifacts/
│   └── gpt_oss_120b_full/                    # the codegen artifact that OOMs
│       ├── graph_2/                          # logits prefill (OOMs in _main during trace capture)
│       │   ├── main.py                       # 4 MB emitted Python TTNN with 403 ttnn.typecast calls
│       │   ├── utils.py                      # DeviceGetter, load_tensor (FABRIC_1D, needs RING patch — Bug 2)
│       │   ├── __init__.py
│       │   ├── run                           # bash wrapper: env + python3 main.py
│       │   ├── ttir_cpu.py
│       │   ├── ttnn.mlir                     # final TTNN dialect MLIR (6.2 MB)
│       │   └── irs/                          # all MLIR lowering stages: vhlo → shlo → ttir → ttnn
│       └── graph_3/                          # logits decode (untested — graph_2 OOM exits the run first)
│           └── ... (same layout, ttnn.mlir is 3.4 MB)
└── regular_path_irs/                         # production-path MLIR for the SAME test, NO codegen
    ├── README.md
    ├── graph_2_logits_prefill/               # 8 IRs (vhlo, shlo, ttir, ttnn variants, etc.)
    ├── graph_3_logits_decode/                # 8 IRs (same layout)
    └── shared/                               # 2 shlo_compiler_cleaned, one per graph
```

**graph_0** (perf prefill) and **graph_1** (perf decode) are intentionally omitted — they're `return_logits=False` clones of graph_2 / graph_3 produced because `benchmark_llm_torch_xla` `torch.compile`s the model twice (once for warmup/timing without logits-D2H, once for accuracy with logits). They share the same compute path, so g2/g3 are sufficient evidence.

**Tensorbins** (the 27 → 693 `.tensorbin` files per graph, 218 GB each — 36 layers' worth of `bf16` weight snapshots) are NOT included. They're the source-of-data inputs the emitted `from_device → typecast(bfp_bf4) → to_device` chain pulls from at runtime; not needed for diagnosing the allocation-order bug, only for actually re-running the emit on hardware.

## How to use this for diagnosis

### Option A — compare IRs without re-running anything

The key files for bisecting where the lowering diverges:

```bash
# Codegen-path TTNN (final pre-emit MLIR for logits prefill, the graph that OOMs)
ls codegen_artifacts/gpt_oss_120b_full/graph_2/irs/ttnn_*.mlir
# Compare to:
ls regular_path_irs/graph_2_logits_prefill/ttnn_*.mlir

# Per the user note (the one that overturned my initial dtype-drop hypothesis):
#   "fastest reproducer would be running ttmlir-opt --ttir-to-emitpy-pipeline
#    on the captured MLIR and dumping IR between passes (--mlir-print-ir-after-all)
#    to see where the chain disappears."
ttmlir-opt --ttir-to-emitpy-pipeline --mlir-print-ir-after-all \
  codegen_artifacts/gpt_oss_120b_full/graph_2/irs/ttir_*.mlir 2>&1 \
  | tee diag.log
```

Things to look for in the diff:
- `memory_config` choices on intermediate tensors — does codegen pick INTERLEAVED DRAM where production picks L1 / sharded?
- `ttnn.deallocate` placements — are there missing deallocate calls in the codegen path that production has?
- `ttnn.to_memory_config` re-layouts — does the codegen path leave tensors in a layout that needs more contiguous space?
- The point_to_point op (and reduce_scatter / all_gather neighbors) — these were the surface symptom (`720 × 4096` intermediate) and live in the sparse-MoE expert dispatch path.

### Option B — re-run the OOM yourself

Requires the full 872 GB artifact (the tensorbins this bundle does NOT carry). Steps:

1. Check out the branch:
   ```bash
   git checkout dgolubovic/autosearch-mp-gpt-oss-120b
   ```
2. Re-emit the codegen on hardware (~12 hours, ~872 GB) — or ~6 hours, ~436 GB after commit `40e715090` on the same branch which skips the perf_wrapper compile:
   ```bash
   docker exec --user $(id -u):$(id -g) \
     --workdir /home/dgolubovic/repos/tt-xla \
     tt-xla-ird-$USER bash -lc '
       source venv/activate && cd tests/benchmark && \
       export CODEGEN_EXPORT_PATH=/some/path/codegen_artifacts/gpt_oss_120b_full
       mkdir -p $CODEGEN_EXPORT_PATH/..
       python -m pytest -svv \
         test_llms.py::test_gpt_oss_120b_tp_galaxy_batch_size_64 \
         --accuracy-testing
     '
   ```
3. Run the harness on the emitted artifact:
   ```bash
   docker exec --user $(id -u):$(id -g) \
     --workdir /home/dgolubovic/repos/tt-xla \
     tt-xla-ird-$USER bash -lc '
       source venv/activate && \
       python tests/benchmark/scripts/run_codegen_decode.py \
         /some/path/codegen_artifacts/gpt_oss_120b_full \
         --graphs graph_2
     '
   ```
4. The OOM hits in `_main` after `from_device → typecast` of the first few layers. The harness handles the FabricConfig.FABRIC_1D_RING and DeviceGetter-shared-mesh workarounds automatically (Bugs 1 and 2 from CODEGEN_BUGS.md).

## What is NOT the bug (already ruled out)

My initial hypothesis was that `apply_weight_dtype_overrides` was being silently dropped from the emit, since `grep to_dtype graph_2/main.py` returned zero. That was a **wrong needle** — the op name in this pipeline is `ttnn.typecast`. Correct counts:

- `main.py` has **403 ttnn.typecast calls**: 108 → `BFLOAT4_B` (= the experts.{gate_up,down}_proj + router globs), 73 → `BFLOAT8_B`, plus intermediates.
- MLIR pipeline preserves the dtype info end-to-end (108 `ttcore.weight_dtype` arg attrs at SHLO, 222 typecast ops at TTIR, 403 at TTNN).
- Pattern is `from_device → typecast(target) → deallocate(host bf16) → to_device` — the same chain `HOST_BFP_PACKING_VALIDATION_NOTES.md` describes for the production path.

So the dtype-override path is healthy. The OOM is something else — allocation order / intermediate-buffer lifetimes — and that's what this bundle is shaped to diagnose.

## Related artifacts on the branch (not bundled here — too large or environment-specific)

- `autoresearch_logs/codegen_artifacts/gpt_oss_120b_full/graph_*/tensors/` — the 870 GB of `.tensorbin` weight snapshots (regenerate by re-running step 2 above)
- `autoresearch_logs/codegen_full120b.log` (266 KB) — the 12-hour codegen wall log that produced this artifact
- `autoresearch_logs/harness_full120b_iter0.log` (13 KB) — the harness invocation that hit the OOM in graph_2/`_main` and exited with `AUTORESEARCH_EXIT=1`
- `autoresearch_logs/sanity_postcommit.log` (254 KB) — clean rerun of the test WITHOUT codegen confirming the production path still works at TOP1=81.25%
- `tests/benchmark/scripts/run_codegen_decode.py` — the harness used to reproduce the OOM
- `CODEGEN_BUGS.md` — full writeup of all three codegen bugs found in this investigation (1: dry_run=False import shadowing, 2: FABRIC_1D vs Topology.Ring mismatch, 3: this OOM)
