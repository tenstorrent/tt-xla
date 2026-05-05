# Hybrid Streaming — Progress (snapshot 2026-05-05, vanilla torch-xla compatible)

Snapshot of the hybrid (Mode 1 + Mode 2) streaming work for DeepSeek-V4-Flash / DeepSeek-Pro on a 512 GB host.

Companion deep-dive: [DEBUG_HYBRID_NOTES.md](DEBUG_HYBRID_NOTES.md) — full investigation log.

## TL;DR

- **Compiler fix is the load-bearing patch.** Disabling `TTNNConstEvalInputsToSystemMemory` (via `STREAM_HYBRID_DISABLE_CONSTEVAL_TO_HOST=1`, plumbed through PJRT compile-options) keeps const-eval function inputs on DEVICE storage so `ensure_layout` actually migrates host→device on first use and the per-shard `DistributedHostBuffer` host data is freed via RAII. Validated end-to-end on 43 layers (full_model_run.log): prefill RSS dropped to ~6 GB, prefill ids deterministic across runs.
- **Per-layer dummy execute + the compiler fix together** are the working release path. The dummy execute triggers SPMD consolidation + ensure_layout; the compiler fix makes ensure_layout actually free host data.
- **Vanilla torch-xla is now supported** (no patched wheel needed). The plugin's `BufferInstance::fireDoneWithHostBufferEvent()` helper, called from `PjrtTensor::ensure_layout` right after `toLayout` migrates host→device, fires the on_done callback that vanilla torch-xla's `kImmutableUntilTransferCompletes` semantics relies on. Without it, the framework holds the source `at::Tensor` alive until BufferInstance teardown (= model lifetime), defeating per-layer streaming. Verified: vanilla wheel + 5-layer hybrid gives flat per-layer RSS (post-flush 4.87 → 5.09 GB across 5 layers); deterministic prefill ids match patched-mode output. Mode 2 (per-layer execute) also passes.
- **Compile-time memoization (Listener-based) lands a >10× speedup on `ttir_to_ttnn`.** `ttcore::valueTracesToConstantArgs` is consulted from many pattern-match attempts inside `TTIREraseInverseOps`, and each call walks the entire use-def chain → O(num_ops²). A new `ConstevalForwardAnalysis` (forward propagation, RewriterBase::Listener for invalidation) drops the pass to O(num_ops) per rewrite epoch. Measured at 10 layer: TTIREraseInverseOps 55.84 s → 2.37 s (~23×); end-to-end ttir_to_ttnn 57 s → 5.34 s. Output is bit-for-bit identical to the unmemoized baseline.
- **Ship-time migration (single-device only) was tried and currently disabled.** See "Reverted / parked" below.

## Reproduce — current best run

```bash
source venv/activate
STREAM_HYBRID_DISABLE_CONSTEVAL_TO_HOST=1 \
STREAM_NUM_LAYERS=43 \
python -u streaming/run_hybrid.py > streaming_log/full_run.log 2>&1
```

No torch-xla env var needed; no patched wheel needed. Plugin's `fireDoneWithHostBufferEvent` is a no-op when the on_done event isn't set (= patched torch-xla path with `kImmutableOnlyDuringCall`), so the same binary works on either wheel.

For 5-layer / 10-layer sanity: replace `STREAM_NUM_LAYERS=43` with the smaller number. Validated 43-layer baseline (full_model_run.log): ship ~28 min, prefill cold compile+exec ~36.5 min, decode 1 cold ~36 min, decode hot ~4 s/token. Total to first 2 tokens: ~100 min.

## Patches applied

### tt-xla (this repo, branch `sshon/aknezevic/ds4-streaming-temp`)

| File | What |
|------|------|
| [pjrt_implementation/inc/api/compile_options.h](../pjrt_implementation/inc/api/compile_options.h) | New `enable_const_eval_inputs_to_system_memory` option (default true, mirrors compiler default) |
| [pjrt_implementation/src/api/compile_options.cc](../pjrt_implementation/src/api/compile_options.cc) | Parses option from PJRT compile-options dict |
| [pjrt_implementation/src/api/module_builder/module_builder.cc](../pjrt_implementation/src/api/module_builder/module_builder.cc) | Forwards to `TTIRToTTNNCommonPipelineOptions::enable_const_eval_inputs_to_system_memory` |
| [pjrt_implementation/inc/api/buffer_instance.h](../pjrt_implementation/inc/api/buffer_instance.h) | Declares `fireDoneWithHostBufferEvent()` |
| [pjrt_implementation/src/api/buffer_instance.cc](../pjrt_implementation/src/api/buffer_instance.cc) | Implements `fireDoneWithHostBufferEvent()`; ship-time migration hook in `copyFromHostBuffer` (gated, single-device only) |
| [pjrt_implementation/src/api/tensor.cc](../pjrt_implementation/src/api/tensor.cc) | `ensure_layout` calls `fireDoneWithHostBufferEvent()` on every shard right after `toLayout(retain=false)`; debug counters and helper env switches retained from investigation |
| [pjrt_implementation/src/api/flatbuffer_loaded_executable_instance.cc](../pjrt_implementation/src/api/flatbuffer_loaded_executable_instance.cc) | Execute-path RSS prints (`[execute-rss …]`) |
| [streaming/run_hybrid.py](run_hybrid.py) | Hybrid driver. Per-layer load+ship+dummy-execute, mutable-KV re-zero after dummy, periodic K-layer release execute, IR dump option, BG decode pre-compile (env-gated), step timing |
| [streaming/run_layer_stream.py](run_layer_stream.py) | Mode-2 helpers reused by hybrid |
| [streaming/run_streaming.py](run_streaming.py) | Mode-2 driver |
| [streaming/DEBUG_HYBRID_NOTES.md](DEBUG_HYBRID_NOTES.md) | Investigation log |
| [streaming/BG_DECODE_PRECOMPILE.md](BG_DECODE_PRECOMPILE.md) | Background decode pre-compile design + open questions |
| [streaming/HYBRID_PROGRESS.md](HYBRID_PROGRESS.md) | This file |

Logs accumulate under `streaming_log/`. Per-experiment naming: `exp{A,B,…}_<scenario>.log`.

### tt-mlir (`third_party/tt-mlir/src/tt-mlir`, branch `aknezevic/ds4-streaming`)

The local `tt-mlir` HEAD has only the ship-time-migration runtime API additions on top of upstream — no further changes are required for the vanilla-compatible streaming path. Working tree is clean as of this snapshot.

| File | What |
|------|------|
| `runtime/include/tt/runtime/runtime.h` | Public API: `bool isTensorOnHost(Tensor)`, `Tensor migrateHostTensorToDevice(Tensor, Device)` |
| `runtime/include/tt/runtime/detail/ttnn/ttnn.h` | TTNN-side declarations |
| `runtime/lib/runtime.cpp` | `DISPATCH_TO_CURRENT_RUNTIME` dispatchers (TTNN impl real; TTMetal/Distributed `fatalNotImplemented`) |
| `runtime/lib/ttnn/runtime.cpp` | TTNN impl: `ttnn::Tensor::to_device(&meshDevice)`, default DRAM/INTERLEAVED |

Additive only — no existing API surface changed. Used by the plugin's single-device ship-time migration path; gated to no-op for multi-device meshes (= our 4×8 production setup), so currently unexercised but kept for future work.

## Reverted / parked

### Pass=ON leak fix (this session, all reverted)

Several attempts to make the default compiler path (pass=ON) work without `STREAM_HYBRID_DISABLE_CONSTEVAL_TO_HOST=1`:

- v2: deallocate input runtime tensors after submit
- v3: relax plugin safety check on `isTensorOnHost`
- v4: reset `m_runtime_tensor` to default-constructed
- v5: migrate host→device after submit
- v6: tt-mlir `debug_apis` storage-type relax
- v7: in-place `ttnn::Tensor` storage reset (`TTNNTensorWrapper::releaseStorage`)
- v8: skip safety check via `host_released` flag
- v9: LoadCachedOp uses unvalidated lookup for version extraction

Conclusion: the GlobalTensorCache key is `(deviceId, programHash, inputVersions)`, where `programHash` is per-compile module. Dummy compile and prefill compile produce different program hashes for the same const-eval function, so cross-compile cache hits don't occur. Without cache hit on prefill the const-eval function actually executes; if the source has been emptied, it crashes. All v2–v9 plugin / runtime patches reverted; `STREAM_HYBRID_DISABLE_CONSTEVAL_TO_HOST=1` remains the production path.

### Multi-device ship-time migration

The right hook is post-`from_pjrt_buffers` consolidation, not in `copyFromHostBuffer` per-shard. Single-device gate ships in this branch but is unused on our 4×8 mesh. Pending design.

## Outstanding work

1. **Persistent compile cache** (across process restarts). For the 43-layer setup, prefill + decode cold compile together still cost tens of minutes even after the memoization win; persisting compile artifacts on disk would let the second run skip both. Touches tt-mlir compile cache + flatbuffer hashing + plugin invalidation policy. Top priority for production usability.
2. **Next compile-time hot pass** is `TTIRQuantDequantConversion` (~40 s at 43 layer) — same kind of investigation pattern as the EraseInverseOps fix.
3. **Multi-device ship-time migration** as described above.
4. **Background decode pre-compile** ([BG_DECODE_PRECOMPILE.md](BG_DECODE_PRECOMPILE.md)) — patch validated to fail with current torch_xla (dynamo `_ModuleStackTracer` not thread-safe across the same nn.Module). Path forward needs `torch.export` / AOT or upstream fix; gated default-off.
5. **DEBUG_HYBRID_NOTES.md cleanup** — env-var knobs and debug prints to remove once path is stabilized.

## Time / RSS measurements (43 layer, full_model_run.log)

| Phase | Time | RSS |
|-------|------|-----|
| Skeleton + top-level ship | 51 s | → 103 GB |
| KV buffer pre-ship | 3 s | → 105 GB |
| Per-layer cycle (43 layers) | ~27 min | flat (~106 GB) |
| Prefill cold compile+exec | 36.5 min | 106 → 6.57 GB |
| Decode cold compile+exec | 36 min | (briefly used) |
| Decode hot exec | 3.93 s/token | stable |
| **Total to 2nd token** | **~100 min** | |

5-layer vanilla torch-xla baseline (expK21):
- post-top-level rss = 4.53 GB; per-layer post-flush 4.87 → 5.09 GB (drift +0.22 GB / 5 layers); prefill 42.7 s; decode 1 cold 40.75 s.

Mode 2 (3-layer, expK22) also passes on vanilla; per-layer RSS oscillates without accumulation.

## Repo state at snapshot

| Repo | Branch | HEAD | Modified |
|------|--------|------|----------|
| tt-xla | `sshon/aknezevic/ds4-streaming-temp` | (this commit) | 7 files in pjrt_implementation/streaming/ |
| tt-mlir | `aknezevic/ds4-streaming` | (local commit on top of `c77d6f9f4`) | 4 files in runtime/ |
