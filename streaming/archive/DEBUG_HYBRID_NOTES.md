# Hybrid streaming — temporary debug instrumentation

Tracks code added to investigate the per-layer host-RAM leak in
`streaming/run_hybrid.py`. Remove or adjust everything listed here once
the leak is understood and a permanent fix is chosen.

All temporary code is tagged with the marker comment:

- C++: `// DEBUG_HYBRID_LEAK`
- Python: `# DEBUG_HYBRID_LEAK`

`grep -rn "DEBUG_HYBRID_LEAK"` from repo root finds every entry.

## C++ side

### [pjrt_implementation/src/api/tensor.cc](../pjrt_implementation/src/api/tensor.cc)

1. **`ensure_layout()` retain=false patch** (around lines 117-134).
   Pass `retain=false` to `tt::runtime::toLayout`, then re-set
   `setTensorRetain(true)` on the new device tensor. Intent: trigger
   the `deallocateTensor` branch in `runtime/lib/ttnn/runtime.cpp`'s
   `toLayout` so the host source is freed after migration.
   *Status*: empirically had ~no effect on RSS. Probably needs a
   different fix (per-shard `m_pjrt_tensor` consolidation), but
   keeping in case it matters under different conditions. **Decision
   needed before merge: keep or revert.**

2. **`ensure_layout()` debug counter** (around lines 85-112).
   Gated by env var `TTPJRT_DEBUG_ENSURE_LAYOUT=1`. Counts total /
   early-return / migrated calls with cumulative migrated_volume.
   Prints to stderr every 50 calls. **Remove before merge.**

3. **Headers added**: `<atomic>`, `<cstdio>`, `<cstdlib>` (for the
   debug counter). If counter is removed and not used elsewhere,
   remove these too.

## Python side

### [streaming/run_hybrid.py](./run_hybrid.py)

1. **`STREAM_HYBRID_SKIP_FLUSH` env var**. When `1`, skips the
   per-layer dummy flush entirely. Used for ablation: compare RSS with
   and without dummy. **Remove or convert to permanent flag once
   leak is resolved.**

2. **`STREAM_HYBRID_TOUCH_ALL_PARAMS` env var**. When `1` (default),
   the per-layer dummy graph explicitly sums every parameter and
   buffer in the block, forcing dynamo to lift them all as graph
   inputs. Intent: force consolidation / host release for params
   that the natural `block.forward` path doesn't reach. **Decision
   needed: if it works, this becomes permanent.**

3. **Extra `_log(...)` calls**: `pre-torch-compile`,
   `post-torch-compile`, `post-prefill-call-pre-sync`. Used to
   localize where exactly the 70 GB RSS drop happens. **Remove
   before merge** (keep just the existing `pre-prefill` /
   `post-prefill`).

## Logs

All experiment logs go under [streaming_log/](../streaming_log/).
Naming convention: `hybrid_<N>layer_<variant>.log`. Examples so far:

- `hybrid_5layer_v2.log` — patched plugin, full default path
- `hybrid_2layer_debug.log` — TTPJRT_DEBUG_ENSURE_LAYOUT=1, NUM_LAYERS=2
- `hybrid_2layer_v3_split.log` — plus extra _log call points
- `hybrid_2layer_skipflush.log` — STREAM_HYBRID_SKIP_FLUSH=1

## Experiment log — first-execute host release investigation

**Confirmed facts** (from prior runs):

| Run | l1 post-flush | post-prefill | drop |
|---|---|---|---|
| NUM_LAYERS=2 no release | 131.10 GB | 63.40 GB | -67.7 GB |
| NUM_LAYERS=5 no release | 176.47 GB | 108.85 GB | -67.6 GB |
| NUM_LAYERS=2 release-at-2 + final prefill | 131.22 → 63.53, then no further drop | 63.44 GB | first call: -67.7 GB; second call: 0 |
| NUM_LAYERS=5 K=2 release | 131.22 → 63.53 (1st works); 93.16 → 93.57 (2nd no drop) | 108.97 GB | first only |
| NUM_LAYERS=5 double prefill | 176.48 → 108.85 (1st), 108.85 → 108.85 (2nd) | - | first only, drop is fixed-size ~67.6 GB |

**Key conclusions so far**:
1. Host RAM "release" happens exactly once per process — at the FIRST big execute.
2. Drop amount is **fixed ~67 GB** regardless of how many layers were shipped (2 → 67, 5 → 67).
3. Subsequent executes (different end_id, different shape, fresh inputs) do NOT release.
4. dynamo cache reset has no effect.
5. ensure_layout's `early=1` for big stacked expert tensors → no migration → my retain=false patch can't fire deallocateTensor for them.
6. `ttnn::deallocate` for HostStorage is a NO-OP ([tensor.cpp:142-143](../third_party/tt-mlir/install/tt-metal/ttnn/core/tensor/tensor.cpp#L142-L143)).
7. Host data release relies on RAII chain when wrapper destructs.
8. Mode 2 works because BufferInstances die per iter → multi-device PjrtTensor's last ref drops → wrapper destructs → host freed via RAII.
9. Hybrid keeps BufferInstances alive forever via `model.layers[i]._parameters` → no destruction → no release.

**Fixed 67 GB drop hypothesis** (unproven): some process-level cleanup at first compile/execute. Candidates:
- ttnn-metal `enable_program_cache` allocation forcing glibc arena compaction
- ttnn::MeshDevice::create allocating large device DRAM, side-effect on host pages
- TT plugin's first openMeshDevice
- Compilation pipeline allocating temp host buffers and freeing them post-compile

### Experiment results

- **Exp A**: ❌ **KV pre-ship is NOT the cause**. Skipping persistent_bufs pre-ship gives essentially identical RSS pattern: l1 post-flush 131.10 (vs 131.10), post-prefill 63.40 (vs 63.40). post-init-buffers is ~5 GB lower without pre-ship but absorbed by first-execute drop. Log: [expA_2layer_skip_preship.log](../streaming_log/expA_2layer_skip_preship.log).
- **Exp B**: ❌ **program cache is NOT the trigger**. With `TT_RUNTIME_ENABLE_PROGRAM_CACHE=0`: l1 post-flush 130.97, post-prefill 63.26 — same drop. Log: [expB_2layer_no_progcache.log](../streaming_log/expB_2layer_no_progcache.log).
- **Exp C-prelude**: ❌ **mesh open via tiny graph is NOT the trigger**. Tiny `(8,8) + (8,8) sum` execute before ship: post-pre-open-mesh 1.41 GB. 67.7 GB drop still happens at final prefill. Log: [expC_pre_open_mesh.log](../streaming_log/expC_pre_open_mesh.log).
- **Exp C-2**: ❌ **embed-only pre-execute is NOT the trigger**. Running `model.embed(ids).sum()` before ship: post-embed-pre-execute 103.12 GB (only +0.1 GB). 67.7 GB drop still happens at final prefill. Log: [expC2_embed_pre_execute.log](../streaming_log/expC2_embed_pre_execute.log).
- **Exp D**: ❌ **glibc arena compaction is NOT the trigger**. With `MALLOC_TRIM_THRESHOLD_=0`: post-prefill 63.37 — same drop. Log: [expD_malloc_trim_threshold.log](../streaming_log/expD_malloc_trim_threshold.log).
- **Notable observation**: post-top-level RSS = **103 GB** even before any layer ship. ~70 GB of that is C++ allocations (Python tensor inventory at l0 post-load shows 26 GB of Python tensors). The first-execute drop releases ~40 GB from this state plus ~28 GB layer staging = 68 GB total.
- **Exp E**: ✅ **Localized the drop to `getInputRuntimeTensors` (`prepareInputTensor` → `from_pjrt_buffers`)**. Added per-call RSS prints inside `FlatbufferLoadedExecutableInstance::execute` (gated by `TTPJRT_DEBUG_EXECUTE_RSS=1`). Logs: [expE_submit_internals.log](../streaming_log/expE_submit_internals.log), [expE2_5layer_release_internals.log](../streaming_log/expE2_5layer_release_internals.log).

## Final mental model

Per-layer ship leaves N per-shard `m_pjrt_tensor` host copies alive (one per device). When an execute calls `prepareInputTensor`, `from_pjrt_buffers` checks `have_same_tensor(shards)`:
- **First time**: shards have separate per-shard PjrtTensors → falls into `from_runtime_tensor` → creates a multi-device PjrtTensor via `createMultiDeviceHostTensor` (which copies HostBuffer shared_ptrs) → `setPjrtTensor(new)` on each shard → OLD per-shard PjrtTensor's last ref drops → `~PjrtTensor` → `~ttnn::Tensor` → `~HostBuffer` → if shared_ptr ref count goes to 0, the underlying `std::vector` is destructed → memory freed back to glibc.
- **Subsequent times**: shards already share multi-device PjrtTensor → `have_same_tensor` true → `from_shards` returns existing → no consolidation → no per-shard destruction → no host data released.

The reason the **first** big graph triggers a ~67 GB RSS drop (vs. small drops for per-layer dummies):
1. First big graph compiles a large new HLO that allocates large temporary host buffers during compile.
2. The compile-time allocations exceed glibc's arena thresholds → glibc compacts arenas → returns previously-freed-but-stuck pages to OS.
3. Subsequent compiles either cache-hit (no fresh alloc) or are smaller (don't exceed thresholds).

**Per-layer dummy doesn't trigger compaction** because:
- Same-cr layers cache hit at PJRT level → no fresh large compile.
- Each new compile is small (single block), under glibc's compaction threshold.

**Subsequent partial-whole-model release execute (release-at-4) doesn't trigger** because:
- Most inputs already consolidated by per-layer dummies + release-at-2 → `have_same_tensor` cache hits → no per-shard destructions.
- Net new consolidation is small (just newly shipped layers' inputs) → glibc isn't pressured into another compaction.

## Implications for DeepSeek Pro on 512 GB host

- Each layer ship: ~14 GB host accumulation (per-shard + multi-device staging).
- Hybrid never frees this until first big execute. After that, no further releases.
- 43 layers × 14 GB ≈ 602 GB peak host before first execute fires → exceeds 512 GB → OOM during ship.
- No batch-and-release strategy fits because subsequent releases don't trigger.

### Exp G: skip per-layer dummy + K=2 release

Hypothesis: per-layer dummy pre-consolidates each layer's shards (cache hit at release time), so by release-at-K most layers are already consolidated and there's nothing fresh for glibc compaction to free. Test: skip dummy entirely.

Result: ❌ **release-at-4 still 0 GB drop** even with no per-layer dummy. So per-layer dummy isn't the cause of subsequent releases failing. The real factor is **size of fresh consolidation**:
- release-at-2: top-level (~5 GB) + layers 0,1 (~28 GB) + KV → enough new multi-device PjrtTensor creation to pressure glibc into arena compaction → **66 GB drop**
- release-at-4: only layers 2,3 (~28 GB) fresh; top-level already consolidated by release-at-2 → **insufficient new allocation** → glibc not pressured → **0 GB drop**

Log: [expG_skip_dummy_K2.log](../streaming_log/expG_skip_dummy_K2.log).

### Exp H attempt: PjrtTensor::force_migrate_to_device

Tried implementing a method that takes a host-resident multi-device PjrtTensor and migrates it via `ttnn::Tensor::to_device(meshDevice)`. The reassignment to `m_runtime_tensor` would destruct the OLD host wrapper, releasing `distributed_host_buffer`'s shared_ptr refs to per-shard host data (RAII).

**Blocker**: requires either (a) plugin to include tt-mlir's detail headers + ttnn/tt-metal full include surface (CMake change adds compile errors due to missing tt-metal common headers), or (b) tt-mlir runtime to expose new public APIs (`isTensorOnHost`, `migrateHostTensorToDevice`) that wrap the ttnn-internal call.

(a) requires ~10+ tt-metal include paths to plugin's CMake; risky for include conflicts and namespace bleed.
(b) requires editing tt-mlir source + rebuilding via ExternalProject — heavier work but cleaner.

Method left as stub in [pjrt_implementation/src/api/tensor.cc](../pjrt_implementation/src/api/tensor.cc) (`force_migrate_to_device`) — declared in [tensor.h](../pjrt_implementation/inc/api/tensor.h) and gated by `TTPJRT_AUTO_MIGRATE_AFTER_CONSOLIDATE` env var, but currently no-ops. To activate, add the two functions to tt-mlir runtime/include/tt/runtime/runtime.h with TTNN-side impl using `to_device`.

### Why force_relayout (Exp F) didn't fix it

`TTPJRT_FORCE_RELAYOUT=1` makes `ensure_layout` always call `toLayout` regardless of `hasLayout`. Empirically:
- l0/l1 dummies dropped ~1.5 GB each (vs ~0 baseline) — small improvement
- release-at-4 still 0 GB drop

Reason: `convertTensorLayout` for matching layout (host→host) returns a ttnn::Tensor that **shares `tensor_attributes` shared_ptr with the source**. Reassigning `m_runtime_tensor` creates a new wrapper but the underlying shared_ptr ref count for the data remains alive (now held by the new wrapper). OLD wrapper destructs, but the data shared_ptr stays alive in the new wrapper. **No actual release.**

To force a true release of host data for HOST-expected inputs, we'd need:
1. Force migration HOST → DEVICE_DRAM (changes storage type) — requires constructing a DEVICE Layout, but Layout construction relies on `LayoutDesc` (private to tt-mlir runtime, not in plugin's include path).
2. Use `ttnn::Tensor::to_device(meshDevice)` directly — requires meshDevice access from PjrtTensor (plumbing) and assumes the compiled graph can accept device input where host was expected (likely breaks correctness).
3. Modify `from_host_shards` to MOVE not COPY HostBuffer shared_ptrs — tt-metal change.
4. Change compiler so executable expects DEVICE storage for layer weights — tt-mlir compiler change.

**Hybrid cannot fit DeepSeek Pro on 512 GB host with the current PJRT plugin / ttnn-metal release mechanism.** Solving this requires either:
1. **Plugin-level: explicit host-only release API** that drops `m_pjrt_tensor`'s `distributed_host_buffer` while keeping the device-resident multi-device tensor alive. Requires either:
   - A new ttnn API to convert host-resident multi-device tensor to device-only (without going through ensure_layout's hasLayout check), then drop host portion.
   - OR a new tt_metal API to compact ttnn-metal's host buffer pool periodically.
2. **Runtime-level: glibc-independent host pool**. Use jemalloc/tcmalloc with aggressive purging, OR force `MALLOC_ARENA_MAX=1` (perf hit), OR custom allocator for ttnn host tensors.
3. **Architecture-level: device-side weights only** — skip the per-shard `m_pjrt_tensor` host copy entirely, transfer host→device synchronously without staging. Requires changes in PJRT plugin's `BufferInstance::copyFromHostBuffer`.

### Pending experiments
- **Exp B**: `TT_RUNTIME_ENABLE_PROGRAM_CACHE=0` to test if program cache allocation is the trigger.
- **Exp C**: Force mesh close+open between releases to retrigger ttnn-metal init (cost: lose device state, need re-ship).
- **Exp D**: Vary mesh shape per release ((2,4) vs (4,2)) to force getOrCreateMeshDevice to reshape (also loses device state).
- **Exp E**: Profile what specific host allocation happens at first compile to identify exact 67 GB source.

## Recommended next step for resolution

The cleanest end-state fix:

1. **Add to `tt/runtime/include/tt/runtime/runtime.h`** (and src copy at `runtime/include/...`):
   ```cpp
   bool isTensorOnHost(Tensor tensor);
   Tensor migrateHostTensorToDevice(Tensor host_tensor, Device device);
   ```

2. **Implement in `tt/runtime/lib/runtime.cpp`** — dispatch to TTNN runtime.

3. **Implement in `tt/runtime/lib/ttnn/runtime.cpp`**:
   ```cpp
   bool isTensorOnHost(Tensor tensor) {
     auto &wrapper = tensor.as<TTNNTensorWrapper>(DeviceRuntime::TTNN);
     return ::tt::runtime::ttnn::utils::isOnHost(
         wrapper.getTensor().storage_type());
   }

   Tensor migrateHostTensorToDevice(Tensor host_tensor, Device device) {
     auto &wrapper = host_tensor.as<TTNNTensorWrapper>(DeviceRuntime::TTNN);
     ::ttnn::Tensor &source = wrapper.getTensor();
     if (!::tt::runtime::ttnn::utils::isOnHost(source.storage_type())) {
       return host_tensor;
     }
     auto &meshDevice = device.as<::ttnn::MeshDevice>(DeviceRuntime::TTNN);
     ::ttnn::Tensor device_tensor = source.to_device(&meshDevice);
     return ::tt::runtime::ttnn::utils::createRuntimeTensorFromTTNN(
         device_tensor, /*meshEvent=*/std::nullopt, /*retain=*/true);
   }
   ```

4. **Rebuild tt-mlir** so install/lib has the new symbols.

5. **Activate the stub** in [pjrt_implementation/src/api/tensor.cc](../pjrt_implementation/src/api/tensor.cc)'s `force_migrate_to_device` — replace the stub body with the two calls (already commented in the function).

6. **Test correctness**: For inputs whose executable expects HOST storage, ensure_layout would normally migrate them back (device→host). Pair with `TTPJRT_NO_DEVICE_TO_HOST_MIGRATION=1` (also stubbed) to skip that back-migration. Risk: ttnn ops that strictly require HOST input may fail. If it fails: tt-mlir compiler change is the alternative — make compiler emit DEVICE storage for layer weights.

## Cleanup checklist (when leak is fixed)

- [ ] Decide on the `retain=false` patch in `ensure_layout` (keep / revert / move under flag)
- [ ] Remove the `ensure_layout` debug counter
- [ ] Remove `<atomic>`/`<cstdio>`/`<cstdlib>` headers if unused
- [ ] Remove `STREAM_HYBRID_SKIP_FLUSH` env var
- [ ] Decide on `STREAM_HYBRID_TOUCH_ALL_PARAMS` (permanent / remove)
- [ ] Remove extra `_log` call points
- [ ] Delete this `DEBUG_HYBRID_NOTES.md` file
