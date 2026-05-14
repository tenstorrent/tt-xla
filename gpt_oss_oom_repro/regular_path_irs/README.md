# GPT-OSS 120B regular-accuracy-path IR dumps

MLIR intermediate-representation dumps captured during a successful **regular** (non-codegen) `pytest --accuracy-testing` run of `test_gpt_oss_120b_tp_galaxy_batch_size_64`. Intended as a reproduction reference for the codegen-path OOM bug — diff these IRs against the corresponding `<codegen-export>/graph_N/irs/*.mlir` files to find where allocation order / intermediate-buffer lifetime diverges from production.

## Source run

- **Test:** `pytest -svv tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_galaxy_batch_size_64 --accuracy-testing`
- **Hardware:** Galaxy 4×8 (32 Wormhole_b0 devices)
- **tt-mlir pin:** `f8d3bf0e97dee04ea1783b00304b37b48d446c62` (PR #4450, 2026-05-04 uplift)
- **Run id:** `run4386`, captured 2026-05-07 ~10:52–11:24 UTC
- **Result:** PASSED — TOP1=81.25%, TOP5=95.31%, decode 568 ms / token, total wall ~37 min
- **No CODEGEN_EXPORT_PATH set** — pure production path through XLA → StableHLO → TTIR → TTNN → flatbuffer execution

## Layout

Each pytest invocation emits 4 graphs (2 wrappers × 2 phases): perf_wrapper (return_logits=False) and logits_wrapper (return_logits=True) each compiled separately, each producing prefill + decode graphs. The perf_wrapper graphs (g0, g1) are intentionally **not included here** — they're `return_logits=False` clones of the logits ones with the same compute path. Only the logits ones (g2, g3) are kept since they're what the autoresearch loop uses for the OOM repro.

| Subdir | Wrapper | Phase | Shape characteristic |
|---|---|---|---|
| `graph_2_logits_prefill/` | `logits_wrapper` (`return_logits=True`) | prefill | `input_ids: (64, 64)` |
| `graph_3_logits_decode/` | `logits_wrapper` (`return_logits=True`) | decode | `input_ids: (64, 1)` |
| `shared/` | — | — | `shlo_compiler_cleaned_*.mlir` (one per graph, timestamps map to g2..g3 by order) |

## Each per-graph subdir contains 8 MLIR dumps

Captured at successive stages of the lowering pipeline:

| File prefix | Dialect / stage |
|---|---|
| `vhlo_*.mlir` | Versioned StableHLO (frontend output) |
| `shlo_*.mlir` | StableHLO raw |
| `shlo_frontend_*.mlir` | StableHLO after frontend passes |
| `shlo_compiler_*.mlir` | StableHLO after compiler passes |
| `shlo_set_mesh_attr_*.mlir` | StableHLO with `sdy.sharding` mesh attrs |
| `ttir_*.mlir` | TTIR (hardware-agnostic tensor IR) |
| `ttnn_*.mlir` | TTNN dialect |
| `ttnn_runtime_*.mlir` | TTNN with runtime annotations (final pre-emit) |

## How to use for the codegen-OOM diagnosis

The codegen run produces an analogous set of IRs under each `graph_N/irs/` in the codegen artifact dir (e.g., `autoresearch_logs/codegen_artifacts/gpt_oss_120b_full/graph_2/irs/`). The two paths share the same frontend → TTIR; they should diverge at TTNN or `ttnn_runtime` where memory configs / shard specs / deallocation timing get finalized.

Suggested diff:

```bash
# Production-path IR for the logits-prefill graph
prod=gpt_oss_120B_ttn_irs_repro/graph_2_logits_prefill/ttnn_runtime_*.mlir

# Codegen-path IR for the same logical graph (logits prefill — graph_2 in the codegen layout)
codegen=autoresearch_logs/codegen_artifacts/gpt_oss_120b_full/graph_2/irs/ttnn_*.mlir

diff -u <(grep -E "memory_config|deallocate|BufferType|TensorMemoryLayout" $prod | sort -u) \
        <(grep -E "memory_config|deallocate|BufferType|TensorMemoryLayout" $codegen | sort -u)
```

Or feed both through `ttmlir-opt --ttir-to-emitpy-pipeline --mlir-print-ir-after-all` and bisect the pass where allocation chains diverge.

## Related artifacts

- `CODEGEN_BUGS.md` (repo root): writeup of all three codegen bugs found in this investigation
- `autoresearch_logs/codegen_full120b.log`: 12-hour codegen wallclock that produced the OOM-ing artifact
- `autoresearch_logs/harness_full120b_iter0.log`: the harness-on-full-120B run that hit the OOM in `_main` of graph_2 (13.7 KB)
- `autoresearch_logs/sanity_postcommit.log`: clean rerun confirming TOP1=81.25% / TOP5=95.31% on the same test (2026-05-11)

The corresponding flatbuffer outputs of the production path (the "what TTNN actually executes") are in `modules/fb_gpt_oss_120b_tp_galaxy_batch_size_64_bs64_isl128_run4386_g0..g3*.ttnn` — not copied here because they're binary blobs.
