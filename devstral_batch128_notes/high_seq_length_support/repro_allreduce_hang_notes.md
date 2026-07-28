# Devstral `ttnn.all_reduce` hang — minimal isolated repro: run guide + result key

Companion to `repro_allreduce_hang_test.py` (an **UNRUN** proposal). Builds on
`allreduce_collision_analysis.md` in this directory.

## What the repro targets

The full Devstral-123B DP+TP chunked-prefill run on a Blackhole galaxy hangs at
the **first `ttnn.all_reduce` of graph-2's warmup** (`devstral_dptp_test.log:15165`,
`cluster_axis=1, reduce_type=sum`, input `1x1x4096x12288` bf16 — the o_proj TP
reduction). Graph-1's byte-identical all_reduce, including inside a trace capture,
succeeded. At tt-metal runtime this composite op decomposes to
`ReduceScatterDeviceOperation + AllGatherDeviceOperation`; the reduce_scatter
allocates a per-call **intermediate** buffer whose address the fabric ring assumes
is identical across all 8 TP peers.

**Leading (unproven) hypothesis:** graph-1 captures the intermediate allocation
inside a trace; graph-2's eager, program-cache-hit call allocates it fresh. If the
fresh post-trace allocation lands at **divergent addresses across the 8 chips**,
peer writes miss → the receiving semaphore never signals → hang. The spec-only
program-cache hash lets graph-2 reuse graph-1's compiled program (no recompile to
force a fresh symmetric layout), which enables/masks the bug but does not cause it.

## Mesh / topology caveat (important)

- The hang is on **cluster_axis=1 (the 8-wide TP axis)**; DP (axis 0) is orthogonal,
  so the repro only needs **8 chips along axis 1**.
- The test opens the **full `[4,8]` galaxy** (tt-metal `mesh_device` fixture with
  param `(4,8)`) and then **`create_submesh(MeshShape((rows, 8)))`** to carve a
  **contiguous** 8-wide region. `submesh_rows=1` → a pure `[1,8]` TP line;
  `submesh_rows=2` → `[2,8]` (adds a little DP breadth while keeping TP as the
  hang axis).
- **Do NOT** hand-pick non-contiguous device ids (e.g. `TT_VISIBLE_DEVICES=0,4,8,…`).
  Prior work found arbitrary carve-outs **fail at cluster init**. `create_submesh`
  carves a contiguous rectangular sub-region, which is a valid connected topology.

## How to run

The test uses the tt-metal pytest conftest fixtures (`mesh_device`,
`device_params`, `silicon_arch_name`, `function_level_defaults`). It **cannot run
from this notes directory** — copy it into the tt-metal ccl test tree first:

```bash
cd /data/ssalice/temp/tt-xla
TTM=third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal
cp devstral_batch128_notes/high_seq_length_support/repro_allreduce_hang_test.py \
   $TTM/tests/ttnn/unit_tests/operations/ccl/

# Run from the tt-metal root so conftest + the models/ imports resolve.
cd $TTM
export TT_METAL_OPERATION_TIMEOUT_SECONDS=60   # fail fast; also set in-file
pytest -svv \
  tests/ttnn/unit_tests/operations/ccl/repro_allreduce_hang_test.py \
  -k "A_cache_trace_churn and fabric_1d and tp_only_1x8 and links_auto"
```

Run on a **clean, idle** 32-chip galaxy only — a real hang can wedge the mesh and
require a host/card reset. Start with the single `-k` selection above (the primary
hang candidate), then widen. To sweep the full matrix drop the `-k`.

## The variant matrix and how to read it

| Variant | cache | trace capture | churn | final call | role |
|---|---|---|---|---|---|
| `A_cache_trace_churn`     | ON  | yes | yes | eager cache-hit | **primary hang candidate** (mirrors graph-1 traced → graph-2 eager hit) |
| `B_cache_off`             | OFF | no  | no  | eager recompile | **A/B control** (fresh symmetric alloc each call) |
| `C_cache_notrace_churn`   | ON  | no  | yes | eager cache-hit | discriminator: is the trace capture required? |
| `D_cache_notrace_nochurn` | ON  | no  | no  | eager cache-hit | negative control (bare cache hit; matches the all_gather that succeeded) |

Also swept: `num_links ∈ {auto(None), 1}`, `submesh_rows ∈ {1, 2}`,
`fabric_config ∈ {FABRIC_1D, FABRIC_2D}`.

### Discriminating outcomes

- **A hangs, B passes** → collision/asymmetric-intermediate mechanism **confirmed
  causally**; the program-cache reuse is load-bearing. File upstream against
  reduce_scatter intermediate allocation under trace + program-cache hit.
- **A hangs AND C hangs** → the trace capture is *not* required; a plain
  cache-hit + allocator churn is enough. Points at reduce_scatter
  intermediate-allocation symmetry independent of tracing.
- **A hangs, C passes** → the **trace capture** is the essential perturbation
  (trace-baked vs. eager intermediate allocation) — the sharpest confirmation of
  the stated hypothesis.
- **D hangs** → even a bare byte-identical cache hit deadlocks; contradicts the
  logged all_gather counterexample (`devstral_dptp_test.log:15053`) and would mean
  the isolated kernel/fabric hangs at this shape regardless of caching.
- **Nothing hangs (A/B/C/D all pass)** → the hang is **not reproducible in
  isolated SPMD**. This is itself a strong, fileable finding: it means the trigger
  needs the model/vLLM allocation pattern or the full `[4,8]` DP interaction to
  create the cross-chip divergence — see the API limitation below.

## Why an isolated repro may NOT hang (the API limitation to report)

A clean mesh runs **SPMD**: the host issues an *identical* allocation sequence to
every chip, so uniform churn stays **symmetric** across chips. **The standard ttnn
mesh API exposes no way to inject per-chip-divergent allocation.** So Variant A can
pass even if the mechanism is real — the isolated harness structurally cannot
manufacture the asymmetry the hypothesis needs. Treat a pass as a *result*, not a
disproof.

The one asymmetry source that **does** exist on real silicon is **per-chip core
harvesting** (different harvested rows → different `compute_with_storage_grid_size()`
→ divergent buffer layouts across the 8 TP chips). The test logs each chip's
compute grid at setup and dumps the **per-chip buffer address** of every all_reduce
output across the 8 peers (`_log_per_device_addresses`, look for
`uniform=False` / `ADDRESS DIVERGENCE`). That instrumentation is the real
deliverable: it shows *whether the hardware even offers* the asymmetry, and — if a
variant hangs — whether the intermediate/output addresses actually diverged.

> Caveat: the address dump probes the all_reduce **output/input** tensors, which is
> what the ttnn API exposes. The suspect buffer is the reduce_scatter **internal
> intermediate**, which the composite op allocates and frees inside a single call
> and does **not** surface to Python. Confirming intermediate divergence
> definitively still needs a tt-metal-side per-device dump of the reduce_scatter
> intermediate address (noted as the decisive experiment in
> `allreduce_collision_analysis.md`).

## Faithfulness notes / knobs

- **Op:** top-level `ttnn.all_reduce` (semaphore-free composite) — the exact entry
  point `runtime/lib/ttnn/operations/ccl/all_reduce.cpp:36` calls, and the exact
  op that decomposes to RS+AG for this shape.
- **num_links:** the tt-mlir compiler emits the AllReduceOp with `num_links=nullptr`
  (`TTIRToTTNN.cpp:1289`), i.e. **auto-select**. `num_links=None` (the `links_auto`
  case) matches Devstral; `1` is included as a cross-check.
- **topology:** the runtime forces the op to 1D; the test uses `Topology.Linear`.
- **fabric_config:** device-level, set by vLLM (not by the op). Unknown from the
  logs, so both `FABRIC_1D` and `FABRIC_2D` are swept. **If you can confirm which
  fabric config the vLLM Devstral run used, pin it** — a fabric-routing hang would
  depend on it.
- **timeout:** `TT_METAL_OPERATION_TIMEOUT_SECONDS=60` is set in-file (before the
  ttnn import) and should also be exported in the shell.
