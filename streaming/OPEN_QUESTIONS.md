# Open Questions — Streaming Implementation

Live tracker. Each item has: brief description, why it matters, current
status. Close items by deleting them or moving to a "Resolved" section
with the resolution.

## Open

### 1. Does `enable_sparse_mlp` work cleanly per-block?

**Why it matters**: The streaming pipeline calls it on a single `Block`
in [streaming_loader.py](./streaming_loader.py). The function as written
walks `model.named_modules()` of whatever you pass; if it has hidden
assumptions about top-level model attributes (e.g., reads `model.config`
without `config=` kwarg), per-block invocation could silently no-op.

**Status**: Visual inspection of [`tt_torch/sparse_mlp.py:1093`](../python_package/tt_torch/sparse_mlp.py)
suggests the traversal is local. Need to **verify by running on one
block and asserting `block.ffn` is now `A2aSparseMLPWithSharedExperts`**.

### 2. Does `enable_sparse_mlp` peak host RAM at 2× block size?

**Why it matters**: Streaming budget hinges on per-block transient
≤ ~15 GB. If the MoE-rewrite holds both the per-expert original tensors
AND the new stacked tensor simultaneously, peak doubles.

**Status**: Need to instrument with `psutil.rss` around the
`enable_sparse_mlp(block, ...)` call. See recipe in
[MEMORY_BUDGET.md](./MEMORY_BUDGET.md).

### 3. `weight_loader.load_block_state_dict` key prefix

**Why it matters**: The streamed dict's keys are likely
`layers.{N}.attn.wq_a.weight` etc. Calling
`block.load_state_dict(stripped_dict, strict=False)` requires us to strip
the `layers.{N}.` prefix. Need to confirm the exact key format.

**Status**: Inspect output of `weight_loader.load_block_state_dict(0).keys()`
and adjust the `removeprefix` logic in
[streaming_loader.py](./streaming_loader.py) accordingly.

### 4. Top-level params: which need explicit `.to(device)`?

**Why it matters**: `embed`, `norm`, `head` are submodules — calling
`.to(device)` on them moves their internal Parameters. But
`hc_head_fn`, `hc_head_base`, `hc_head_scale` are direct
`nn.Parameter` attributes on `Transformer`, not inside a submodule.
They need to be reassigned via `setattr(model, name, p.to(device))` or
`model._parameters[name] = p.to(device)`.

**Status**: Implementation in [streaming_loader.py](./streaming_loader.py)
should handle both cases. **Verify after first end-to-end run**.

### 5. `torch.compile` cache path with streamed model

**Why it matters**: dynamo identifies the model via internal hash. If
the streaming-loaded model and the all-at-once-loaded model trace to
identical FX graphs, the existing compile cache is reused. Otherwise
streaming forces a fresh compile on every run.

**Status**: Probably identical — streaming changes load order, not
forward graph. **Confirm by checking compile-time logs (`+dynamo`) on
streaming run vs e2e**.

### 6. SPMD `mark_sharding` on per-block params

**Why it matters**: Mark-sharding has to land on the underlying
`nn.Parameter` storage (not a parametrize wrapper). Streaming applies
sharding **before** `apply_weight_dtype_overrides` so this is the simple
case — but verify the per-block sharding loop covers every parametrized
param: `wq_b`, `wo_a`, `wo_b`, `kv_cache`, `mlp.router.gate.weight`,
`mlp.experts.{gate,up,down}_proj`, plus indexer/compressor when present.

**Status**: Mirror [`transformer_shard_spec`](../tests/torch/models/deepseek_v4/test_deepseek_v4_full_e2e.py#L187)
into a per-block `block_shard_spec(block, mesh)` helper.

### 7. Const-eval blob size after streaming

**Why it matters**: Once compiled, the const-eval prologue uploads a
giant tensor of folded constants to each device. Whether this fits in
each device's DRAM is independent of host RAM streaming.

**Status**: Independent of streaming. Documented here only because users
might confuse "streaming worked → device OOM during const-eval" as a
streaming bug. It's not — it's the const-eval blob from the existing
forward graph.

### 8. KV cache buffer creation timing

**Why it matters**: `Attention.__init__` calls `register_buffer("kv_cache", torch.zeros(...))`. When the block is on CPU, this allocates on CPU. After `block.to(device)`, it ships to TT. So peak host RAM during streaming includes one block's kv_cache buffer (~32 MB / block).

**Status**: Negligible (~32 MB) but document for completeness.

### 9. Failure recovery

**Why it matters**: If layer 30's load fails mid-stream, layers 0-29 are on TT but model is incomplete. Currently no resume capability.

**Status**: **Out of scope for v1**. Document as known limitation.

### 10. Logging / progress bar

**Why it matters**: Streaming 43 blocks takes minutes. Users need feedback.

**Status**: Add `tqdm` over the layer loop in
[run_streaming.py](./run_streaming.py). Trivial.

### 11a. Eliminate dynamo re-trace per layer iter (perf, **attempted, parked**)

**Second attempt update (2026-05-01, evening)**: implemented the
*template + in-place `param.data.copy_(...)`* variant — preserves
template module Python identity AND param storage location, so dynamo
guards no longer fire. Verified end-to-end on NUM_LAYERS=10:

- **dynamo cache DOES hit**: the `"Found an argument on non-XLA device"`
  warning count drops 10 → **0** across the layer loop.
- **But trace time is unchanged**:
  ```
  baseline (fresh inst, miss): cr=0 trace 9.92s/8.86s, cr=4 20.7s, cr=128 12.1s
  template (cache hit):        cr=0 trace 10.2s/8.93s, cr=4 21.3s, cr=128 11.5s
  ```
  i.e., dynamo's Python tracing is NOT what dominates `t_trace`. The
  per-call ~9-21s overhead lives downstream — in torch_xla's
  `extract_compiled_graph` dispatch + LTC arg registration for the
  ~24-34 lifted params/buffers per block. Cache-hit only saves the
  cold-vs-warm delta of ~1-2s.
- **Plus device DRAM OOM** at NUM_LAYERS≥10: 3 templates persistent on
  device + per-layer kv_cache buffers exceeds 8-device 2x4 mesh DRAM
  budget. Repros at bsz=8 / prompt=128 inside `bank_manager.cpp:462`.

Net: **template approach doesn't help**. The flag (`STREAM_TEMPLATE=1`)
is left in place for future experiments but defaults off.

Repro: [`_repro_inplace_copy_cache.py`](./_repro_inplace_copy_cache.py)
(small-scale shows partial savings 12.3s → 8.1s, but those savings
come from PJRT compile-cache hit, NOT dynamo, and don't translate to
real run because PJRT cache already hits across layers in the existing
flow).

**First attempt update (2026-05-01, earlier)**: tried the
`template + torch.func.functional_call` pattern with both
*block-as-arg* and *closure-captured template* variants. dynamo still
re-traced every layer iter (warning fired). Per-cr templates kept
alive caused device DRAM OOM at NUM_LAYERS=4. Same conclusion: net
loss. See `_repro_functional_call_cache.py`.

**Where the real overhead lives** (open question): `t_trace` is the
time of `run_block(...)` returning. With dynamo cache hit, this enters
`XLAExecutor._call_experimental_compile`, which:
1. Iterates args and force-moves CPU tensors to XLA (should be no-op
   for already-XLA inputs).
2. Calls cached `compiled_graph(*params_and_consts + args)` — a
   torch_xla `bridge.extract_compiled_graph` artifact.
3. Returns. The PJRT compile cache hits (graph hash matches), so the
   actual execute should be quick.

The 9-21s suggests step 2 (LTC arg registration + dispatch through
the compiled artifact) is non-trivial for ~30 lifted tensors per
block. Worth profiling with `XLA_IR_DEBUG=1` or a Python profiler to
nail down. Rough hypothesis: each lifted-param XLA tensor needs a
guard-like check (shape/dtype/device) and an LTC node binding before
the compiled fn can run.

Possible paths if revisited:
1. **Profile and optimize torch_xla's per-call overhead** — likely the
   biggest lever, but requires upstream patches.
2. **Bypass dynamo + LTC entirely**: capture StableHLO once per (cr,
   shape), call PJRT execute directly with refreshed param buffers.
   Heavy refactor.
3. **Reduce # of lifted params** by fusing weights upfront (e.g., one
   big concat tensor per layer) to cut the LTC arg-registration loop.

The original analysis (kept for context):



**Why it matters**: Each layer iter creates a fresh `block_i` instance.
`torch.compile`'s dynamo cache keys on the module's weakref, so every
new instance is a cache miss → dynamo re-traces (StableHLO build, etc.)
every layer. The TT-MLIR backend caches by graph hash so the actual
hardware compile only happens once per `(compress_ratio, shape)`, but
the dynamo Python work runs every iter.

**Symptom**: Every layer iter prints
`"Found an argument on non-XLA device ... Force moving the argument
to XLA"` (a dynamo-trace-time warning). Estimated ~3-5s/layer of
hidden trace overhead inside `t_exec`. For NUM_LAYERS=43 ×
MAX_NEW_TOKENS=3 = 215 layer iters this is ~6-10 minutes wall time.

**Approach**: combine the v1 *template + weight swap* idea (archive)
with `torch.func.functional_call` so the cache key is stable:

```python
# Build ONE template block per unique compress_ratio (3 templates total
# for cr=0/4/128). Templates persist; their weights are placeholders.
templates = {cr: build_template_block_for_cr(cr) for cr in (0, 4, 128)}

@torch.compile(backend="tt")
def run_block_fn(template, params, h, sp, ids):
    # Pass params as explicit dict input → cache hits on
    # (template id, shape) regardless of params identity.
    return torch.func.functional_call(template, params, (h, sp, ids))

# Per layer iter:
inst = _build_layer_instance(layer_id, ...)
params = dict(inst.layers[layer_id].named_parameters())
# Ship params + buffer splice
ship_params_to_device(params)
template = templates[args.compress_ratios[layer_id]]
h = run_block_fn(template, params, h, sp, ids)
```

- **v1's template+swap had a memory issue** (it pre-streamed all 43
  layers into 43 device-resident instances, defeating streaming). The
  fix here is: only keep ONE template instance per cr (not per layer);
  weights stream per-iter via the `params` argument.
- Need to verify dynamo cache key behavior for `functional_call` —
  some prior probing in [`_repro_functional_call_cache.py`](./_repro_functional_call_cache.py).
- Persistent buffers (kv_cache etc.) still need to be spliced into the
  template's buffer slots before each iter (or passed via functional
  call too — TBD which is cleaner).

**Status**: Designed; not implemented. **Priority over #11b** because:
- (a) saves wall time independent of pipelining, and
- (b) makes pipelining cleaner — the loader thread can prepare a
  `params` dict without touching template module state.

---

### 11b. Pipeline next-layer load behind current-layer execute (perf, **landed**)

**Update (2026-05-01)**: implemented in `run_layer_stream.py` via a
single-worker `concurrent.futures.ThreadPoolExecutor`. Default ON,
disable with `STREAM_PIPELINE=0`.

Measured impact on NUM_LAYERS=4 / bsz=8 / prompt=128 / 1 prefill + 1
decode (no PCC):
- Baseline: s0=199s, s1=201s, total ~400s
- With pipeline: s0=150s, s1=140s, total ~290s
- **~25-30% wall time reduction**, primarily from per-layer load
  dropping from ~32s to ~8-10s (CPU build is now hidden behind the
  previous layer's exec).

Correctness preserved: PCC=0.997-1.000 across layers, identical token
output to baseline. Host RAM peak rises from ~65 GB to ~110 GB
(2 instances briefly resident), still well within 503 GB.

Original analysis kept below for context.



**Why it matters**: Each layer iter currently runs sequentially as
`load (~12s) → exec (~12-20s)`. The CPU side of `load` (HF read +
dequant + sparse_mlp rewrite, ~10s of the ~12s) is fully independent
of the current layer's device execution and can run in parallel.

For NUM_LAYERS=43 × MAX_NEW_TOKENS=5 = 215 layer iterations, hiding
~10s/iter behind exec saves ~36 minutes per full prefill+decode run.

**Scope**: 1-step lookahead only. A bounded-queue producer-consumer
(thread + Queue with depth K) was considered but rejected — load and
exec are already balanced once `INLINE_PCC=0`, so K>2 buys robustness
to load-time variance but no throughput gain, and the design adds
moderate complexity.

**Sketch (1-step lookahead, CPU-build only)**:
```python
import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
inst_curr = _build_layer_instance(0, ...)
for layer_id in range(NUM_LAYERS):
    fut_next = (executor.submit(_build_layer_instance, layer_id + 1, ...)
                if layer_id + 1 < NUM_LAYERS else None)
    # splice + ship + exec on inst_curr
    ...
    inst_next = fut_next.result() if fut_next else None
    del inst_curr; inst_curr = inst_next
```

- Host RAM peak: ~2 instances × ~13 GB = ~26 GB. We have 503 GB. Fine.
- Device DRAM unaffected (only CPU build is overlapped).
- GIL: HF read is I/O, dequant/sparse_mlp is mostly torch ops that
  release the GIL, so contention should be limited.

**INLINE_PCC interaction**: when `STREAM_INLINE_PCC=1`, the CPU eager
forward is `O(seq=128, 256 experts × 6 active)` ≈ 35s/layer and
*depends on the current `h`*, so it can't be prefetched and dominates
load time. In INLINE_PCC mode pipelining only saves the ~12s build
phase (~15-20% wall time). Real pipelining payoff is in production
mode (no PCC) where load≈exec≈15-20s and overlap ≈ doubles throughput.

**Status**: Designed; not implemented. Defer until current full-run
verification passes; then implement Phase 1 first.

## Resolved

### 2026-05-05: hybrid streaming end-to-end + vanilla torch-xla compatibility

- **Hybrid mode** (`run_hybrid.py`): per-layer load+ship+dummy-execute on a persistent skeleton, then whole-model compile for prefill+decode. Validated on full 43 layers (`streaming_log/full_model_run.log`): ship ~28 min, prefill cold ~36.5 min, decode 1 cold ~36 min, decode hot ~4 s/token, post-prefill RSS ~6 GB. See [`HYBRID_PROGRESS.md`](./HYBRID_PROGRESS.md).
- **Compiler fix** (`STREAM_HYBRID_DISABLE_CONSTEVAL_TO_HOST=1`) is the load-bearing flag — disables the `TTNNConstEvalInputsToSystemMemory` pass so `ensure_layout` actually migrates const-eval inputs host→device.
- **Vanilla torch-xla supported**. The plugin's new `BufferInstance::fireDoneWithHostBufferEvent` (called from `PjrtTensor::ensure_layout` after `toLayout`) fires the `kImmutableUntilTransferCompletes` on-done callback so vanilla path also drops framework `at::Tensor` after host→device migration. No env var or patched wheel needed at run time. Verified on 5-layer hybrid (`expK21_clean_5layer.log`) and 3-layer Mode 2 (`expK22_mode2_3layer.log`).
- **Leak-fix attempts on the pass=ON path** (v2–v9) reverted: GlobalTensorCache key includes per-compile `programHash`, so cross-compile cache hits don't occur and source-release breaks subsequent executes. Notes archived in [`archive/DEBUG_HYBRID_NOTES.md`](./archive/DEBUG_HYBRID_NOTES.md).

### #2: `enable_sparse_mlp` does pin ~13 GB / block (verified)

Confirmed via `psutil.rss` measurements with `malloc_trim(0)` to
defeat allocator pooling: `enable_sparse_mlp` keeps the original 256
expert tensors alive via two refs:
- `mlp._original_mlp` (stashed via `object.__setattr__`)
- `mlp.experts.original_experts` (registered `nn.ModuleList`)

Streaming-mode workaround: `_strip_cpu_golden_refs(block)` in
[streaming_loader.py](./streaming_loader.py) breaks both refs after
`enable_sparse_mlp` and before `block.to(device)`. With strip, the
sparse-rewrite step no longer doubles per-block memory (verified at
NUM_LAYERS=4).

### #11: ✅ Resolved — `block.to(xla:0)` retention path identified and bypassed

Originally listed as a blocker. After deep investigation, the
streaming mechanism works correctly with three pieces in place:

1. **Use `_xla_tensors_from_aten` (handle path)** instead of
   `nn.Module.to(device)` to avoid `XLATensor::tensor_data` shadow.
   Implemented in [`streaming_loader.py::_ship_module_handle_path`](./streaming_loader.py).
2. **Patch torch_xla `pjrt_computation_client.cpp`** to use
   `kImmutableOnlyDuringCall` semantics so the source CPU `at::Tensor`
   isn't pinned by the `BufferFromHostBuffer` on-done lambda. Patch
   applied at [`pjrt_computation_client.cpp:283-293`](/localdev/sshon/ws/pytorch/pytorch-xla/torch_xla/csrc/runtime/pjrt_computation_client.cpp#L283).
   Built-from-source torch_xla wheel installed.
3. **Drop tensor-keyed dicts after building id-keyed lookup** —
   `_block_shard_spec` returns `Dict[Tensor, Tuple]`; the keys hold
   strong refs to the OLD CPU Parameters. `del block_specs` after
   constructing `block_specs_by_id` so per-block CPU tensors aren't
   pinned by the spec dict.

After all three, the only remaining host RAM during loading is the
TT PJRT plugin's "owned host tensor" (one copy per buffer in
[`buffer_instance.cc:222`](../pjrt_implementation/src/api/buffer_instance.cc#L222)
under `kImmutableOnlyDuringCall`). This staging is held by the
`BufferInstance`'s `m_pjrt_tensor` field and gets released the first
time the buffer participates in execution — `ensure_layout` in
[`tensor.cc:79-88`](../pjrt_implementation/src/api/tensor.cc#L79)
replaces `m_runtime_tensor` with a device-resident version, dropping
the host copy.

**Verified by repro** [`_repro_drop_after_execute.py`](./_repro_drop_after_execute.py)
with a 4 GB BF16 tensor uploaded sharded:
| Phase | RSS |
|---|---|
| baseline | 1.11 GB |
| after cpu tensor created | 5.40 |
| after send (sharded, with input_sharding) | 9.70 |
| after del t | 5.41 |
| **post-sync (compile+execute)** | **1.17 (full release)** |

**Implication for full 43-layer streaming**:
During the load loop the per-block transient stacks. With BF16
stacked experts ~13 GB / block and no compile/execute mid-stream,
total transient ≈ N × 13 GB. For NUM_LAYERS=43 that's ~570 GB host
RAM peak — too much for typical hosts, but releases on the first
prefill execute.

Mitigations for the full-model case (each handles the same
underlying issue from a different angle):
- **Periodic synthetic execute every K blocks** (cheapest): force
  `ensure_layout` to fire mid-stream so prior blocks' host stagings
  release. K~5 keeps the transient bounded around ~65 GB.
- **Eager BFP4/BFP8 materialization per-block**: apply
  `apply_weight_dtype_overrides` inside the per-block loop and
  materialize the quantized form before upload. ~4-8× host transient
  reduction.
- **TT PJRT plugin patch**: in `BufferInstance::copyFromHost` after
  `createOwnedHostTensor`, immediately push to device and replace
  `m_runtime_tensor` so the host owned tensor is released without
  waiting for `ensure_layout`. Requires care around layout — the
  device-side layout is determined by the compiled program, so an
  eager push may need a default layout that gets re-laid-out later.

For NUM_LAYERS=4-5 (current dev workflow) the transient sits around
~65 GB which is fine on the dev box. Streaming task **complete** as
scoped; full-43-layer host RAM optimization is a follow-up.

## Old Blockers (resolved)

### #11 (original wording, kept for context): `block.to(xla:0)` does NOT release CPU storage in torch_xla LTC

**Symptom**: per-block RSS grows linearly by ~13 GB even after
`block.to(device) → torch_xla.sync(wait=True) → gc.collect() → malloc_trim(0)`.

**Confirmed real memory growth (not allocator pooling)** at NUM_LAYERS=4:
| Phase | RSS | VmS | system used | system avail |
|---|---|---|---|---|
| post-top-level-ship | 4.4 GB | 76 GB | 58 GB | 484 GB |
| post-block-0-ship | 17.6 GB | 76 GB | 72 GB | 469 GB |
| post-block-1-ship | 30.7 GB | 77 GB | 85 GB | 456 GB |
| post-block-2-ship | 43.9 GB | 77 GB | 98 GB | 442 GB |
| post-block-3-ship | 57.1 GB | 77 GB | 112 GB | 429 GB |

System used grows ~13 GB / block — **the data is genuinely held in
RAM**, not just allocator pooling. Projected at NUM_LAYERS=43:
~570 GB. **No memory savings vs all-at-once.**

**What we tried**:
| Mitigation | Effect |
|---|---|
| `_strip_cpu_golden_refs(block)` (drops `_original_mlp` + `original_experts`) | Helps inside `enable_sparse_mlp` (avoids ~13 GB doubling within a block) but doesn't change the per-block tail |
| `torch_xla.sync(wait=True)` after each ship | No effect on RSS |
| `gc.collect()` after each ship | No effect (no Python cycles) |
| libc `malloc_trim(0)` | No effect (memory really is live) |
| `_set_xla_enable_device_data_cache(False)` | No effect — that flag controls a different cache (probably hash → tensor lookup) |
| Explicit `param.data = …to(device)` | Equivalent to `.to(device)` via `_apply` |

Block params correctly report `device='xla:0'` post-ship.

**Verified root cause** (read torch_xla source 2026-04-30):

The retention is **not** from `DeviceData` IR nodes. It's from the
`XLATensor::data()->tensor_data` field — a CPU tensor "shadow" that
torch_xla intentionally keeps alive after upload as a convenience
cache for `ToTensor()` calls. Citation:
[`xla_graph_executor.cpp:708-715`](/localdev/sshon/ws/pytorch/pytorch-xla/torch_xla/csrc/xla_graph_executor.cpp#L708):
> "we uploaded the at::Tensor data to the device, but such data is
> still valid so we leave it live on the XLA tensor (so that a
> following ToTensor() does not need to fetch it from device)."

Path that hits this retention:
1. `block.to(xla:0)` → `XLATensor::Create(cpu_tensor, device)` →
   stores `data()->tensor_data = cpu_tensor` ([`tensor.cpp:63-70`](/localdev/sshon/ws/pytorch/pytorch-xla/torch_xla/csrc/tensor.cpp#L63))
2. `torch_xla.sync(wait=True)` → `CollectSyncTensors` → uploads via
   `CreateTensorsData` and sets `data()->handle` — but **does not
   clear `data()->tensor_data`** ([`xla_graph_executor.cpp:714`](/localdev/sshon/ws/pytorch/pytorch-xla/torch_xla/csrc/xla_graph_executor.cpp#L714)).

The CPU tensor stays alive for the lifetime of the XLA tensor. For a
13 GB block with 256 expert tensors stacked, that's the entire 13 GB
held forever.

**Verified workaround path: `_xla_tensors_from_aten`**

`torch_xla._XLAC._xla_tensors_from_aten(tensors, devices, shardings)`
takes a different path:
- Calls `CreateTensorsData(at_tensors, shardings, devices)` which —
  in SPMD mode with a sharding spec — calls
  `ShardingUtil::ShardTensor(...)` to physically split the CPU tensor
  into per-device shards before upload
  ([`tensor_util.cpp:856-869`](/localdev/sshon/ws/pytorch/pytorch-xla/torch_xla/csrc/tensor_util.cpp#L856))
- The resulting handle is wrapped via the **handle-based**
  `XLATensor::Create(handle)` constructor
  ([`tensor.cpp:72-79`](/localdev/sshon/ws/pytorch/pytorch-xla/torch_xla/csrc/tensor.cpp#L72)),
  which sets only `data()->handle` and **does not set `tensor_data`**.
- After the BufferFromHostBuffer transfer-complete lambda fires
  ([`pjrt_computation_client.cpp:285`](/localdev/sshon/ws/pytorch/pytorch-xla/torch_xla/csrc/runtime/pjrt_computation_client.cpp#L285)),
  the AtenSource ref drops and CPU storage is released.

Public Python wrapper exists:
`torch_xla.core.xla_model.send_cpu_data_to_device(data, device, input_sharding)`
([`xla_model.py:1272`](/localdev/sshon/ws/pytorch/pytorch-xla/torch_xla/core/xla_model.py#L1272)).

**Plan for streaming v2**:

1. Build the model on `device="meta"` (no CPU allocation for params).
2. For each block:
   a. Load that block's state_dict to CPU.
   b. Apply `enable_sparse_mlp` + `_strip_cpu_golden_refs` on CPU.
   c. For each Parameter on the block:
      - Build a `ShardingSpec(mesh, partition_spec)` matching the
        per-block shard plan.
      - Call `_xla_tensors_from_aten([cpu_tensor], [device_str],
        [spec.xla_spec(cpu_tensor)])` to upload + shard in one step.
      - Wrap result as `nn.Parameter` and `setattr` it on the
        submodule.
      - Drop the CPU tensor reference.
   d. After all params for the block are uploaded, drop the
      block-local state_dict and force `torch_xla.sync(wait=True)`.
3. For top-level params (embed/head/norm/hc_*), use the same path
   with replicated sharding (or no sharding, which gets the
   implicit-replicated SPMD path).

This sidesteps `nn.Module.to(device)` entirely for streamed weights
and should keep the per-block transient at ~13 GB (single block in
flight) without the lingering shadow.

Two open subquestions:
- Does `torch.compile`'s dynamo trace still recognize handle-only
  `XLATensor`s as `nn.Parameter` inputs? They lack `tensor_data` but
  have `_parameters` registration on the module — should be fine.
- For non-sharded top-level params, the implicit-replicated path
  creates 8 shallow at::Tensor copies sharing one storage; once all
  8 lambdas fire, storage drops. Verify with NUM_LAYERS=4.

**Value delivered so far**:
- Working scaffolding for per-block load (`streaming_loader.py`).
- Verification that `enable_sparse_mlp` keeps duplicate refs and a
  fix for that (`_strip_cpu_golden_refs`).
- Hard data identifying the real blocker as torch_xla
  `XLATensor::tensor_data` shadow retention (not Python GC, not
  allocator pooling, not DeviceData IR).
- Concrete next-step path: `_xla_tensors_from_aten` /
  `xm.send_cpu_data_to_device`.
- Proof that compile is single-pass: NUM_LAYERS=2 prefill fires
  exactly one `SyncTensorsGraph.NNNN` (matches the user's
  expectation of one compile per hidden-layer pattern).

---

## ✅ Major milestone — Layer-streaming inference working (NUM_LAYERS=43, 2 tokens)

**Date**: 2026-04-30

End-to-end full DeepSeek-V4 (43 layers) layer-streaming inference
verified on 8-device dev box. See [`run_layer_stream.py`](./run_layer_stream.py).

**Architecture**:
- Top-level params (embed/head/norm/hc_*) — device-permanent.
- Persistent kv_cache buffers per layer — device-permanent (small).
- Per-iter: build fresh CPU model instance with HF weights → splice
  persistent device kv_cache buffers in → ship parameters → execute →
  capture mutated kv_cache state → del instance + `_round_trip(h)` to
  break IR chain.

**Numbers**:
- 1 token = ~27-31 minutes (43 layers × ~40s/layer).
- Step 0: cold compile (3 variants: cr=0, cr=4, cr=128) + execute.
- Step 1: cache hits across token boundary (exec ~3-6s/layer pure
  device time).
- Device DRAM bounded: ~265 MB/bank steady state (1 layer + top-level
  + 43 small kv_caches).
- Host RSS swells per-token (~430 GB peak) but drops back to ~10 GB
  at end of token (head() compute triggers cleanup).

**Mechanisms identified**:
1. `del temp_inst + del h_out + del h_in + gc.collect + sync +
   wait_device_ops` — releases device buffers.
2. `_round_trip(h, device)` — `h.cpu().to(device)` materializes h
   to a fresh handle, breaking IR chain back into the just-executed
   block (which would otherwise pin its weights).
3. Persistent kv_cache device buffers — splice into each fresh CPU
   block's `_buffers` so attention state propagates across token
   steps and across the temp-instance lifecycle.

**Cache hit cross token boundary** — step 1 layers run without
recompilation:

```
[infer s1 l 0 cr=  0] total=25.11s load=20.71s exec=3.34s
[infer s1 l 1 cr=  0] total=34.13s load=30.49s exec=2.58s
```

The 3 variants compiled in step 0 are reused throughout step 1.

**Decoded** (small token quality known issue):
```
'How are you today?' -> ' math the'
'What is the capital of France?' -> ','
```

**Layer-wise PCC** infrastructure: `streaming/pcc_utils.py`. Two modes:
- `STREAM_INLINE_PCC=1` runs CPU eager forward in parallel with the
  device path inside a single iteration; immediate per-layer PCC.
- `STREAM_REF_MODE=capture|compare` saves/loads activation files
  (per-step, per-layer) for golden-trace comparison.

**Pending follow-up**:
- Pipelining: while layer N executes (~5s), prefetch layer N+1 weights
  in background thread (~37s load is bottleneck). Could halve
  per-token latency to ~15 minutes.
- Investigate dynamo `xla_args` / tt backend `params_and_consts`
  retention — current implementation leaks ~10 GB host per layer
  within a token, partial release at token end. Cleaner release would
  avoid the 430 GB transient peak.
