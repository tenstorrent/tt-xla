# Streaming — archived material

Prior prototypes + investigation logs that are no longer load-bearing
but kept for reference.

## Investigation logs

| File | Notes |
|---|---|
| [`DEBUG_HYBRID_NOTES.md`](./DEBUG_HYBRID_NOTES.md) | Debug instrumentation log from the per-layer host-RAM leak investigation. Most of the C++/Python `DEBUG_HYBRID_LEAK` markers it tracks have either been retained as production-useful env switches (e.g. `TTPJRT_DEBUG_ENSURE_LAYOUT`) or removed; the document itself is obsolete after the leak was resolved by the compiler-fix flag (`STREAM_HYBRID_DISABLE_CONSTEVAL_TO_HOST=1`) plus the plugin's `BufferInstance::fireDoneWithHostBufferEvent`. See [`../HYBRID_PROGRESS.md`](../HYBRID_PROGRESS.md). |

## Prior `run_layer_stream.py` iterations

These are the iterations that led to the canonical
[`../run_layer_stream.py`](../run_layer_stream.py). Kept for reference
in case any of these strategies become useful again (e.g. if the device-
DRAM release semantics change, or if we want to compare wall-times).

| File | Strategy | Why superseded |
|---|---|---|
| `run_layer_stream_v1_template_swap.py` | One persistent compiled `template = model.layers[0]`. Pre-stream all N layers into N parallel `m_i` instances on device. Per-iter swap `template._parameters` ← `m_i.layers[i]._parameters`. | Holds **all N layers' weights on device simultaneously** — only avoids host-side N× peak; device DRAM still scales with N. Defeats the purpose for the 43-layer case. Also relies on dynamo cache hit by-shape rather than by-handle, which is fragile. |
| `run_layer_stream_v2_persistent_skeleton.py` | One persistent skeleton with all kv_cache buffers shipped once. Per-iter stream-load layer i's HF weights → ship to skeleton.layers[i] → execute → replace `_parameters` with meta-device tensors. | Replacing `_parameters` with meta tensors does **not** release device DRAM — `XLATensor::tensor_data` shadow retention keeps the buffers alive. Device DRAM monotonically grew across layers. |
| `run_layer_stream_v3_fresh_instance.py` | Build a fresh model instance per iter (load weights, ship). Run block. Round-trip `h.cpu().to(device)` to break IR chain. `del fresh, h, h_out` + `gc.collect` + `wait_device_ops`. | Releases device DRAM correctly! But **kv_cache state is lost** between iterations because it lives inside the disposed instance. Only valid for prefill-only, single-step runs. |
| `run_layer_stream_v4_hybrid.py` | Persistent skeleton (for kv_cache) + temp instance per iter (for weights). Per-iter ship temp's `_parameters` *into the skeleton* → execute → del temp, round-trip h. | Worked correctly but accumulated device DRAM more slowly than v2 because `_parameters` replacement on the skeleton still triggered the same shadow-retention issue intermittently. v5's external buffer dict made the kv_cache lifetime explicit and side-stepped the skeleton entirely. |

The canonical approach (v5 → `run_layer_stream.py`) keeps:
- v3's fresh-instance + del + round-trip (proven device-DRAM release)
- v4's idea of decoupling kv_cache from weights, but pushed further:
  kv_cache buffers are stored in an **external dict of device tensors**
  and spliced into each fresh instance's `_buffers` map.

See [`../OPEN_QUESTIONS.md`](../OPEN_QUESTIONS.md) for the device-DRAM
release investigation that produced the v3 → v5 progression.
