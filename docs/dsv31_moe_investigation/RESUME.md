# DeepSeek-V3.1 / GLM4 galaxy MoE low-PCC investigation — RESUME doc

Issue #5096 (Grace Engelage). Benchmark: `tests/benchmark/test_llms.py::test_deepseek_v3_1_tp_galaxy_4_layers`
(Wormhole **galaxy, 32 chips**, mesh `(4,8)` axes `("batch","model")`, opt_level=0, bsz=64,
prefill ≈ **16 tokens** tiled across the batch, `experimental_weight_dtype=bfp_bf8`).

Model path: `third_party/tt_forge_models/deepseek/deepseek_v3_1/pytorch/modified_modeling_deepseek.py`
(NOT modeling_deepseek.py). first_k_dense_replace=3 (layers 0-2 dense, layer 3+ sparse MoE:
256 experts, top-8, n_group=8, topk_group=4, sigmoid noaux_tc, routed_scaling_factor=2.5).

The MoE runs on `tt_torch/sparse_mlp.py::A2aSparseMLP` (all_to_all_dispatch → sparse_matmul ×3 →
all_to_all_combine → reduce_scatter). All ops are custom (`torch.ops.tt.*` in `tt_torch/custom_ops.py`).

---

## TL;DR — there are (at least) 2 independent problems

### ✅ BUG #1 — `all_to_all_dispatch` drops half the tokens (FOUND + FIXED)
Root cause: the 2D RING/TORUS fabric makes tt-metal `get_num_links` **overcount routing planes**
(returns **4**, only **2** usable per cluster axis). With num_links=4 the dispatch's sender cores 2,3
(the 2nd half of tokens) send on **non-existent routing planes** → their tokens are silently dropped.
Net: **half the batch (mesh rows 2,3) gets zero routed output**.

Proof (no device needed, run `docs/dsv31_moe_investigation/scripts/verify_dispatch_drop.py` against a
diverse op-dump): dispatch metadata is exactly 50% populated, only 512/1024 tokens survive, combine
output for the dropped half ≈ 0.

**FIX = force `FABRIC_1D`** on any 32-device galaxy (this is PR #5094's approach, which gated it to
Blackhole; we extended it to Wormhole). Applied in `client_instance.cc::computeFabricConfig`.
- Verified: routed-only **seq=128 tiled 0.70 → 0.98**; `num_links` goes 4 → 1.
- Equivalent confirmation: forcing `A2A_NUM_LINKS=2` on the 2D fabric also gives 0.98 (num_links=4 → 0.70).
- **Proper upstream fix belongs in tt-metal `get_num_links` (moe_utils.cpp)** — it should not sum both
  axis directions / should return *usable* planes per cluster axis. FABRIC_1D is the tt-xla workaround.

### ❌ PROBLEM #3 — outlier-magnitude token collapse (OPEN, the benchmark ceiling)
FABRIC_1D fixes seq=128 but the **benchmark (seq=16 tiled) is unchanged** (4L 0.978 / 6L 0.887).
Decomposing the seq=16 frozen MoE block (`routed_shared_split` test):
- routed PCC **0.19**, shared 1.00, block 0.358.
- The killer is **one outlier token** (s=8: hidden absmax 7.5, **CPU routed absmax 2.69 = the largest**,
  but **device routed 0.008** → 333× collapse). Other tokens are small/noisy.
- fp32 algorithm (FORCE_DEVICE_PATH) combine = **2.73** (preserves outlier); bf16 device combine = **0.11**.
  So it's a **bf16-vs-fp32 collapse of outlier-magnitude routed tokens**.
- Every individual op verifies **1.0 vs its own fp32 golden** → the collapse happens at the FIRST op that
  loses the outlier, and downstream ops look "correct" because they consume the already-collapsed value.
- This is the original issue note's **"routing concentration + outlier magnitude"** data effect.

**Why not pinned yet:** `all_to_all_dispatch` **permutes/sparsely places** tokens (dispatched `flat[8]`
is an EMPTY slot; the outlier is elsewhere), so manual per-token tracing through the tiled/sharded sm
dumps is unreliable. Need to **de-permute via metadata** to trace the one outlier token gate→act→down on
device vs a co-located fp32 reference → that pins the exact op.

**Likely fix (hypothesis, unverified):** keep the routed-path **intermediate activation in fp32**
(sparse_matmul output dtype is bf16; the large activation between the two sparse_matmuls is stored bf16).
fp32_dest_acc is already ON (accumulation is fp32) — the issue is the *storage* dtype between ops.

### Other things RULED OUT (don't re-chase)
- gate selection / topk tie-break (handoff's old thread): **refuted** — frozen (identical) routing still fails.
- sub-tile flat tiling (seq<32 packs 2 batch-slots/tile): real for *diverse*, but `PAD_SEQ_TILE` padding
  did NOT move the tiled benchmark (tiled rows identical → no mixing).
- expert_mapping 1D-vs-2D: no effect (reverted).
- bf16 dispatch/combine/sparse_matmul kernels: all verified **1.0** (bit-exact vs fp32).
- math_fidelity/fp32_dest_acc: already HiFi4 + fp32_dest_acc in the compiled graph (precision maxed).
- The benchmark emits gibberish but that's EXPECTED for a 4-6 layer truncated model; PCC is the metric.

---

## Number map (all tiled unless noted, FABRIC_1D on)
| test | seq=16 | seq=32 | seq=128 |
|---|---|---|---|
| routed_only frozen | 0.83 | 0.96 | 0.98 |
| moe_block real | 0.36 | — | 0.98 |
| benchmark 4L / 6L | 0.978 / 0.887 | — | — |

Per-layer ~0.96-0.98 (seq≥32) is bf16 storage; 6L compounds ~0.96³≈0.90 < 0.96 threshold.

---

## EXACT CHANGES per repo (branch `sshon/dsv31-moe-pcc-investigation` in each)

### tt-xla (the real fix + tests + debug knobs)
- **`pjrt_implementation/src/api/client_instance.cc`** — **THE FIX**: (1) open `(4,8)` parent mesh
  directly for 32 devices; (2) `computeFabricConfig` returns `FABRIC_1D` for any 32-device galaxy
  (env `MOE_FORCE_FABRIC_1D=0` to disable). **Requires plugin rebuild** (see below).
- `python_package/tt_torch/sparse_mlp.py` — env-gated `PAD_SEQ_TILE` seq-padding (sub-tile probe, did
  not help tiled); mesh_shape threading for build_expert_mapping (dormant, mesh_shape not passed);
  `FORCE_DEVICE_PATH`/`SPARSE_DUMP_DIR` `_save` debug hooks.
- `python_package/tt_torch/custom_ops.py` — CPU sparse_matmul flat-fallback for seq<32 (KEEP, real fix
  for a crash); env-gated `SPARSE_BF16` debug.
- `tests/benchmark/benchmarks/llm_benchmark.py` — env-gated `PAD_PREFILL_TILE` (real-token prompt growth
  to a tile multiple; 6L 0.887→0.906 but not a real fix — pad tokens pollute PCC if pad-token version).
- `tests/torch/models/deepseek_v3_1/test_deepseek_v3_1.py` — **all the repro/diagnostic tests** (NEW file,
  untracked). Key tests + env knobs below.
- `docs/dsv31_moe_investigation/` — this doc + the offline analysis scripts.

### tt-metal (DEBUG instrumentation only — NOT the fix; remote is misconfigured, local branch only)
Path: `third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/`
- `ttnn/.../ccl/all_to_all_dispatch/all_to_all_dispatch.cpp` — `A2A_NUM_LINKS` env override of num_links.
- `ttnn/.../ccl/all_to_all_dispatch/device/all_to_all_dispatch_program_factory.cpp` — `[A2AD_DBG]` fprintf
  of mesh coord / tokens_per_device / num_links / per-core token ranges.
- `ttnn/.../ccl/all_to_all_dispatch/device/kernels/dataflow/writer_all_to_all_dispatch.cpp` — pre-existing
  zero-init from an earlier session (benign). **These are debug; the real upstream fix is `get_num_links`.**

### tt_forge_models (branch `sshon/dsv31-moe-pcc-investigation`)
- `deepseek/.../loader.py` — `MOE_CLUSTER_AXIS` env knob (default 0).
- `deepseek/.../modified_modeling_deepseek.py` — bias-centering + scatter→one_hot + gather→einsum in
  MoEGate (XLA-compat / routing-preserving; NOT the fix but harmless, keep).

### tt-mlir — no source changes (only read).

---

## REBUILD (needed after pulling the branches)

### tt-xla plugin (for client_instance.cc / FABRIC_1D)
```
source venv/activate
ninja -C build pjrt_plugin_tt.so
# python_package/pjrt_plugin_tt/pjrt_plugin_tt.so is a symlink to build/... → auto-live
```

### tt-metal ttnn (for the A2AD_DBG / A2A_NUM_LINKS debug; ~3 min incremental)
```
cd third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/build_Release
ninja ttnncpp
cp ttnn/_ttnncpp.so lib/_ttnncpp.so
cp ttnn/_ttnncpp.so /localdev/sshon/tt-xla/third_party/tt-mlir/install/lib/_ttnncpp.so   # runtime links build_Release/lib
```
(The runtime `_ttmlir_runtime.so` links `build_Release/lib/_ttnncpp.so` directly — confirm with `ldd`.)

---

## REPRO / DIAGNOSTIC commands

Tests live in `tests/torch/models/deepseek_v3_1/test_deepseek_v3_1.py`. Run on the galaxy.
```
source venv/activate

# the isolated benchmark-gap repro (real routing):
pytest -svv tests/.../test_deepseek_v3_1.py::test_deepseek_v3_1_moe_block_real_input[tiled_prompt-bench_scale]

# routed vs shared split (shows the outlier collapse) — frozen routing, seq via param:
pytest -sq tests/.../test_deepseek_v3_1.py::test_deepseek_v3_1_routed_shared_split[None-64]   # seq=16
SPLIT_SAVE=/path/x.pt pytest -sq ...routed_shared_split[None-64]   # dumps {tt,cp} for offline per-token analysis

# frozen routed-only at chosen seq (ROUTED_SEQ env), tiled:
ROUTED_SEQ=16 pytest -q tests/.../test_deepseek_v3_1.py::test_deepseek_v3_1_routed_only_frozen_routing[tiled_prompt-64]

# per-op device dump (DebugHooks) → tmp/op_dump*; envs:
#   OP_DUMP_DIR, DUMP_DIVERSE=1 (distinguishable tokens), DUMP_BATCH, DUMP_SEQ, DUMP_FROZEN=1,
#   DUMP_METADATA_ONLY=1, ALL_OPS_LOG=<file> (logs every TTNN op asm = the graph)
OP_DUMP_DIR=tmp/op_dump_x DUMP_DIVERSE=1 DUMP_BATCH=64 pytest -q tests/.../test_deepseek_v3_1.py::test_dump_sparse_matmul_io[64]

# num_links / fabric experiments (after the tt-metal debug rebuild):
A2A_NUM_LINKS=2 MOE_FORCE_FABRIC_1D=0 pytest -q ...routed_only_frozen_routing[tiled_prompt-64]

# full benchmark (8 min, 4L; --num-layers 6 for 6L):
pytest -svv tests/benchmark/test_llms.py::test_deepseek_v3_1_tp_galaxy_4_layers
```

Offline analysis (no device) against the dumps, in `docs/dsv31_moe_investigation/scripts/`:
- `verify_dispatch_drop.py` — ROOT-CAUSE PROOF for #1 (run on a diverse dump). Edit the `D=` path.
- `verify_combine_xdev.py` — correct cross-device combine golden (PCC 1.0). Combine gathers along the
  4-row cluster axis → each device gets its **column's 32 experts** (nz=0.125), value = down_all[owner(e),e%8,t].
- `verify_sparse_matmul.py` — sparse_matmul vs fp32 golden from device inputs (PCC 1.0).
- `sparse_cpu_golden.py` — FORCE_DEVICE_PATH algorithm-on-CPU = PCC 1.0 (OOMs at bench scale; nopad/small only).

Dump layout learned the hard way:
- dispatch **permutes/sparsely places** tokens (flat ≠ bd*16+s; many empty slots). De-permute via metadata.
- combine metadata is **token-major, all-gathered** (identical across devices); first B*S rows = original tokens.
- expert e → device `e//8` (1D sequential mapping); down_out slot for token t = t (first B*S of the 4096 BD*S).
- mesh (4,8): linearized = row*8+col; cluster_axis=0 = rows (size 4 = "batch"); reduce_scatter cluster_axis=1
  = columns (size 8 = "model") sums the routed output across the 8 expert-columns.

---

## NEXT STEPS to close #3 (the real remaining work)
1. **De-permuted outlier trace**: in the frozen seq=16 dump, use the dispatch metadata to find the slot of
   the outlier token, then trace gate(sm0)→act(sm2_in)→down(sm2_out)→combine for THAT slot on device, and
   compute the same in fp32 from the dispatched value. Find the first op where device ≠ fp32-algorithm
   (not device-vs-its-own-golden). Need sm dumps in all-shards (`allshards` already includes "sm").
2. If it's the bf16 activation storage: try keeping `SiLU(gate)*up` in fp32 into the down sparse_matmul
   (sparse_mlp `_sparse_expert_forward` / `SparseMLP.forward`), or a tt-mlir change to fp32 intermediate dtype.
3. Land #1 (FABRIC_1D) as its own PR — it's the verified, independent win. Strip the tt-metal A2AD_DBG /
   A2A_NUM_LINKS debug and the tt-xla `PAD_*`/`*_SAVE` knobs before merge.

See also persistent memory: `~/.claude/projects/-localdev-sshon-tt-xla/memory/deepseek-v3-1-moe-pcc.md`.
