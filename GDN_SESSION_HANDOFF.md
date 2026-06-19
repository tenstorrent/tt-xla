# GDN / Qwen3.6-27B vLLM bring-up — session handoff

Branch `mmanzoor/vllm-qwen-3.6`. Repo `/localdev/sgligorijevic/tt-xla`. Single Tenstorrent
Wormhole loudbox (8 devices, ~12 GiB DRAM/device).

> Read order: §1 goal → §2 what's done → §3 current blocker (memory) → §4 the open
> disagreement → §5 facts/evidence → §6 file changes → §7 env/workflow gotchas.
> §5 is measured evidence; hypotheses are explicitly labeled UNVERIFIED.

---

## 1. Original goal
Qwen3.6-27B was up through vLLM on TT but produced incoherent text. The Gated Delta Net
(GDN) port was a "quick and dirty" stub with wrong math. Task: implement a correct GDN
(modeled after FLA / flash-linear-attention), wire it into the TT vLLM plugin
(`integrations/vllm_plugin/vllm_tt/`), and test against FLA reference kernels. Then it
evolved into getting the model to **compile and fit** on the card.

## 2. What was built and verified
- New op package `integrations/vllm_plugin/vllm_tt/layers/gdn/`: `l2norm.py`, `gating.py`,
  `conv1d.py`, `recurrent.py`, `chunk.py`, `attention.py`, `__init__.py`. FLA-faithful,
  pure-PyTorch (no Triton, no `torch.ops.vllm.*`).
- Replaced the wrong `_tt_gdn_core_fallback` (used `sigmoid(a)` gate, never used q/k,
  dropped the `-β(Sk)k` delta term, no conv, no state) — that was the incoherence cause.
- **Math correctness: VERIFIED.** All ops match a brute-force reference at PCC = 1.0 (CPU);
  chunk-prefill == recurrent-prefill == reference. Test harness in
  `tests/integrations/vllm_plugin/gdn/` (`gen_goldens.py` for the GPU box, `test_gdn_ops.py`,
  `_reference.py`, `README.md`). 8/8 op tests pass on CPU.
- **Per-op on card: VERIFIED.** All ops run on the Wormhole card and match CPU at
  PCC ≥ 0.99999 (eager mode), and compile under `torch.compile(backend="tt", fullgraph=True,
  dynamic=False)` with **no graph breaks**.

## 3. Compile failures that were fixed (so the model now compiles past GDN)
The vLLM model compiles via `torch.compile(backend="tt", dynamic=False)`. Initial failures and fixes:
1. `torch.linalg.solve_triangular` (chunk op) is a **CPU fallback** → crashed torch_xla's
   `partition_fx_graph_for_cpu_fallback` (`AssertionError`, or SIGSEGV standalone). **Fixed:**
   replaced with a matmul-only block-recursive unit-lower-triangular inverse
   (`_inv_unit_lower_tri` in `chunk.py`).
2. **Data-dependent loop trip counts** — `for cs in range(int(cu_seqlens[n]), ...)` — dynamo
   can't unroll; broke compile. **Fixed:** single-sequence (`n_seq==1`) paths derive bounds
   from `.shape` (static under `dynamic=False`).
3. **Host-scalar control flow** `int()/bool()` on tensors in conv/recurrent → graph breaks
   that crash the fallback partitioner in the full multi-layer graph. **Fixed:** branchless
   single-seq paths using `index_select`/`index_copy_`/mask-multiply.
4. `model_runner.py::_build_per_layer_attn_metadata` now populates `has_initial_state`.

**Constraint learned:** ops in the `backend="tt"` graph must avoid CPU-fallback ops AND
data-dependent loop counts / host-scalar (`int()/bool()`) control flow. Eager-on-xla
validation does NOT exercise dynamo and hides these — always test with
`torch.compile(backend="tt", dynamic=False)`.

## 4. Depthwise conv1d — known TT bug, worked around (NOT the current blocker)
The GDN token-mixing conv is depthwise (`groups == channels`). `F.conv1d(groups=conv_dim)`
→ a single `ttnn.conv2d(groups=10240, 1×35 image, 1×4 kernel)` which **fails on Wormhole**:
- C=10240: fast compile (~0.8 s) then `TT_FATAL: DRAM Auto slice could not find valid slice
  configuration` (`ttnn/.../op_slicing/op_slicing.cpp:266`) on an *empty* device.
- small C: dispatches then **hangs on readback** (completion-queue spin).

**Worked around** in `conv1d.py`: depthwise causal conv as K shifted slice-multiply-adds
(elementwise, bf16, no `ttnn.conv2d`). Behind a flag `_USE_NATIVE_CONV1D` (default `False` =
workaround; `True` = the proper conv, currently broken). Full bug report:
**`gdn_depthwise_conv1d_bug.md`** (repo root). Toolchain: tt-mlir `53762683b`, tt-metal `1efd27cafdb`.

## 5. CURRENT BLOCKER: device DRAM OOM, and what's measured

After GDN compiles, the model OOMs during the `num_tokens:1` warmup forward:
```
TT_FATAL: Out of Memory: Not enough space to allocate 21135360 B (21 MB) DRAM buffer across 12 banks
  (allocated: 1060793184 B, free: 9980000 B, largest free block: 1207296 B)   # ~11.78 GiB/12 GiB used
```
The 21 MB is trivial; the card is ~99% full. **Question: what fills it?**

### Measured (TT_RUNTIME_MEMORY_LOG_LEVEL=operation, per device, 12 banks × ~1.02 GB):
- Small const-eval probe graph peaks ~15 MB/bank.
- DRAM jumps to **~994 MB/bank ≈ 11.65 GiB/device** "as weights uploaded as program inputs",
  BEFORE the forward graph's first op.
- Forward scratch adds only ~13 MB/bank (~0.15 GiB).
- **KV cache never allocated** (OOM is in warmup). `KV cache sizing: device DRAM = 12 GiB,
  gpu_memory_utilization = 0.200, KV cache budget = 2.40 GiB` → `GPU KV cache size: 384 tokens`.
- L1 never used (everything in DRAM).
- The 21 MB failing buffer = a `TilizeWithValPadding` feeding a QKV/MLP `ttnn.concat`, output
  `tensor<2560x4120xbf16>`.

### Model config (confirmed, from HF `Qwen/Qwen3.6-27B` text_config):
- **DENSE, not MoE.** 64 layers = 48 `linear_attention` (GDN) + 16 `full_attention`.
- hidden 5120, intermediate 17408, **vocab 248320** (huge), tie_word_embeddings=False.
- GDN: `linear_num_value_heads=48`, `linear_num_key_heads=16`, head dims 128, conv_kernel 4
  → key_dim=2048, value_dim=6144, **conv_dim=10240**.
- Full attn: `num_attention_heads=24`, **`num_key_value_heads=4`**, head_dim=256.
- ~26 B params dense × bf16 ≈ **52 GB total weights**.

### Sharding (mesh / specs):
- Mesh: `xs.Mesh(ids, (2,4), ("batch","model"))` → **batch=2, model=4**. `use_2d_mesh` default True.
- `vllm_distributed_utils.py` specs (per device): big weights via `XlaMergedColumnParallelLinear`
  / `XlaQKVParallelLinear` use `("model","batch")`; `RowParallelLinear` `("batch","model")`;
  plain `ColumnParallelLinear` `("model", None)`; lm_head `("model", None)`; embed `(None,"model")`;
  GDN A_log/dt_bias `("model",)`. KV/GDN-state caches sharded on `"model"` only.

### Ideal vs measured:
- Ideal 8-way TP: 52 GB / 8 ≈ **6.6 GB/device**. Measured resident ≈ **11.65 GB/device** ≈ 1.75×.

## 6. THE OPEN DISAGREEMENT (unresolved — needs the next agent's judgment)

Two interpretations of the 11.65 GB, **neither confirmed**:

- **(A) Weights are under-sharded (~/4 not /8).** First memory subagent's read: weights aren't
  sharded tightly enough for 8 devices. Lever: shard everything /8.
- **(B) Weights are already /8 (~6.6 GB) + ~5 GB const-eval.** My (assistant's) reframe — the
  user said this "doesn't make sense", so treat it skeptically. Reasoning was: on the 2-D mesh
  `("model","batch")` shards KV cleanly (K/V weight `[1024,5120]` → 1024/4 × 5120/2, both divide)
  and should give /8 for all big weights; and `TTConfig` docstring warns const-eval stores
  folded constants on device "essentially the entire model once per [precompiled] graph"
  (tt-mlir issue #3888). **Caveat against (B):** at the FIRST bucket only one graph is compiled,
  and const-eval folds *constant subgraphs*, not necessarily a full model copy — so ~5 GB of
  const-eval is an assumption, not measured.

**Neither subagent cleanly separated "weights" from "const-eval" in the 11.65 GB.** The memory
log attributed it to "program inputs" (const-eval results ARE fed as program inputs, so they may
be lumped with weights).

### Experiments tried re: sharding
- **8-way TP via 1-D mesh** (`use_2d_mesh=False` → mesh `(1,8)`): **cleared the OOM** (no more
  21 MB failure) but introduced a NEW failure: StableHLO sharding error because
  `num_key_value_heads=4` doesn't divide 8 — `safe_mark_sharding` replicate-fallback fires (64×
  "dim 1 (size 4) not divisible by mesh axis 'model' (size 8)") but tt-mlir rejects it:
  `Could not compute local sharded shape for result 6: tensor sharding is incompatible with tensor
  shape` → `Failed to run stablehlo pipeline` / `Error code: 13`. It fails in
  `_precompile_backbone`→`_dummy_run` BEFORE weight upload, so the /8 weight footprint was NEVER
  measured (we don't know if /8 would be ~6.6 GB).
- **User's proposed scheme:** keep 2-D mesh; K/V replicate across the batch-pair (2) and shard
  across model (4); every other tensor shard one dim across BOTH axes `(None,("batch","model"))`
  → /8.
- **tt-mlir source verdict on that scheme:** multi-mesh-axis-on-one-tensor-dim is supported
  ONLY on the **Shardy** path (`ShardyUtils.cpp:489` divides by product of axes); the **GSPMD**
  path (default for torch-xla `mark_sharding`) **hard-fails** — `GSPMDUtils.cpp:446`
  `determineGSPMDShardingDims` requires each sharded dim size == a single mesh-axis size
  (`shardShape[1]=8 != meshShape=4` → "Fail to determine shardDims"). So the scheme needs
  torch-xla switched to emit Shardy AND the ttnn runtime double-`chunk_ndim` (same dim twice,
  `runtime/lib/ttmetal/meshshard_utils.cpp:49-58`) validated.

### Decisive experiments NOT yet run (recommended):
1. **`enable_const_eval=False` on the 2-D mesh, re-measure DRAM.** If it drops to ~6.6 GB →
   interpretation (B), const-eval is the fill, OOM fixed with one flag, no sharding change. If it
   stays ~11.65 GB → interpretation (A), weights really are /4.  ← cheapest disambiguator.
2. If (A): either fix the 2-D `("model","batch")` sharding to actually reach /8, or pursue Shardy
   for the user's double-axis scheme.

## 7. Files changed this session
NEW:
- `integrations/vllm_plugin/vllm_tt/layers/gdn/{__init__,l2norm,gating,conv1d,recurrent,chunk,attention}.py`
- `tests/integrations/vllm_plugin/gdn/{gen_goldens,test_gdn_ops,_reference}.py`, `README.md`, `golden/.gitignore`
- `gdn_depthwise_conv1d_bug.md` (repo root)

MODIFIED:
- `vllm_tt/overrides.py` — GDN override now points at `layers/gdn/attention`.
- `vllm_tt/layers/__init__.py` — re-export from new package.
- `vllm_tt/model_runner.py` — `has_initial_state` in `_build_per_layer_attn_metadata` (~line 634).
- `vllm_tt/vllm_distributed_utils.py` — `safe_mark_sharding` now accepts a tuple-of-axes spec
  entry (combined-size divisibility); added `SHARD_ALL=("batch","model")`. **Partition specs
  themselves are NOT changed** (the double-axis rewrite was NOT applied — gated on the Shardy
  question above). This file is in a consistent state (tuple support is a no-op until used).
- `tests/integrations/vllm_plugin/generative/test_mrope.py` — added `"use_2d_mesh": False` to
  additional_config (the 1-D mesh experiment). **Should be reverted to test the 2-D path / const-eval probe.**

DELETED:
- `vllm_tt/layers/gdn_linear_attn.py` (old wrong-math override).

`conv1d.py` flag `_USE_NATIVE_CONV1D = False` (workaround active). `attention.py` flag
`_PREFILL_USE_RECURRENT = False` (chunked delta rule active; recurrent-prefill toggle exists).

## 8. Environment / workflow gotchas (important)
- **Single-user card.** Never run two card processes at once.
- **Always `source venv/activate`** in each shell (state doesn't persist between commands).
  It sets `PYTHONPATH` so `vllm_tt` resolves to the **source** `integrations/vllm_plugin/vllm_tt`,
  NOT the **stale copy** at `venv/lib/python3.12/site-packages/vllm_tt` (dated Jun 11, has NO
  `layers/gdn/` dir — running against it silently uses old code). It also makes `infra` importable
  for the test conftest. Running `venv/bin/python` directly skips this.
- **Reset:** `venv/bin/tt-smi -r` (`tt-smi` not on PATH). A `kill -9`/timeout-kill of a card
  process leaves device 4 FW in a bad state ("Device 4 init: failed to initialize FW") → reset
  before the next run. Clean exits don't need a reset.
- **Repro run** (in-process engine + faulthandler + per-op memory log):
  ```
  source venv/activate && VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONFAULTHANDLER=1 \
    TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG TT_RUNTIME_MEMORY_LOG_LEVEL=operation \
    timeout 1500 python -X faulthandler -m pytest \
    tests/integrations/vllm_plugin/generative/test_mrope.py -x -s
  ```
  Takes ~5-8 min (loads 27B). `TT_RUNTIME_MEMORY_LOG_LEVEL=operation` emits per-op
  `Device DRAM memory state: MemoryView{...}` snapshots (per device). Latest logs:
  `/tmp/memlog.log` (2-D), `/tmp/memlog_8way.log` (1-D).
- Compile is `DYNAMO_TRACE_ONCE`, `backend="tt"`, `dynamic=False`. py-spy/gdb available
  (`venv/bin/py-spy`, `/usr/bin/gdb`) for hang diagnosis.
