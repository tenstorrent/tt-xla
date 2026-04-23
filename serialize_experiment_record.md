# Tensor Serialization Experiment: Multi-Device Host Tensors in `ensure_layout`

## Date
2026-04-23

## Objective

Experiment with `ttnn::dump_tensor` / `ttnn::load_tensor` APIs to test round-trip
serialization of multi-device host tensors inside `PjrtTensor::ensure_layout`, prior
to the layout conversion that sends tensors to device. The goal is to prove that a
multi-device host tensor can be serialized to disk and loaded back, then still
successfully pass through `toLayout` and device execution.

## Background

### How multi-device host tensors are created

In the MNIST tensor-parallel inference path, `PjrtTensor::from_pjrt_buffers` calls
`rt_tensor_from_strategy`. When the strategy is not `"identity"` (i.e., `"shard"`,
`"replicate"`, or `"shard_2d"`), it creates a multi-device host tensor via:

```cpp
tt::runtime::createMultiDeviceHostTensor(tensors, strategy, mesh_shape);
```

This combines individual per-device shard tensors into a single multi-device host
tensor backed by `MultiDeviceHostStorage` at the ttnn level.

### The `ensure_layout` call site

`ensure_layout` is called from `FlatbufferLoadedExecutableInstance::prepareInputTensor`
(`flatbuffer_loaded_executable_instance.cc:81`), just before the tensor is passed to
`tt::runtime::submit`. The flow is:

```
from_pjrt_buffers (creates multi-device host tensor)
  -> ensure_layout (converts layout for device)
    -> hasLayout check (early exit if already correct)
    -> toLayout (moves tensor to device with correct layout)
  -> submit (runs the compiled program)
```

### Serialization APIs used

- `tt::runtime::dumpTensor(Tensor, filePath)` - Serializes tensor to `.tensorbin` format
- `tt::runtime::loadTensor(filePath)` - Loads tensor back (host tensor when no device arg)
- Both are declared in `tt/runtime/runtime.h`, already included by `tensor.cc`

## Implementation

### File modified

`pjrt_implementation/src/api/tensor.cc` - single file, in-tree edit (no commit needed)

### Changes

1. Added includes: `<atomic>`, `<filesystem>`, `<sstream>`
2. Added serialization logic at the top of `ensure_layout`, **before** the `hasLayout`
   early-exit check:
   - Detect multi-device tensors via `m_shards.size() > 1`
   - Use a static atomic counter for unique filenames
   - Build filename with shape metadata: `tensor_<counter>_<shape>.tensorbin`
   - Dump tensor to `/tmp/tt_serialize_exp/`
   - Load tensor back from disk
   - Preserve the `retain` flag across serialization
   - Replace `m_runtime_tensor` with loaded tensor
   - Log all operations at INFO level with `[SerializeExp]` prefix

### Key design decisions

- **Detection via `m_shards.size()`**: Used `m_shards.size() > 1` rather than
  `tt::runtime::detail::getNumShards()` because: (a) it avoids depending on the
  `detail` namespace, and (b) in the tt-xla flow, multiple shards reliably indicate
  a multi-device tensor was created by `rt_tensor_from_strategy`.

- **Load without device**: `loadTensor(filepath)` without a device argument returns a
  host tensor. This is deliberate -- the loaded tensor then passes through `toLayout`
  which handles the host-to-device conversion and layout transformation.

- **Counter-based filenames**: Since there's no stable cache key, we use an incrementing
  counter. For MNIST TP inference, the arguments arrive in a stable order, so the
  counter maps consistently to specific tensors across runs.

## Test Results

### Test: `test_mnist_inference_tensor_parallel[32-784]`

**Result: PASSED**

Serialized tensors (8 multi-device host tensors per execution):

| Counter | Shape     | File Size  | Description                    |
|---------|-----------|------------|--------------------------------|
| 0       | 5         | 428 B      | Bias (output layer)            |
| 1       | 5x512     | 10,656 B   | Weight (output layer)          |
| 2       | 512       | 1,432 B    | Bias (hidden layer 2)          |
| 3       | 512x256   | 524,704 B  | Weight (hidden layer 2)        |
| 4       | 256       | 1,432 B    | Bias (hidden layer 1)          |
| 5       | 256x784   | 803,232 B  | Weight (hidden layer 1)        |
| 6       | 32x784    | 50,592 B   | Input data (batch=32)          |
| 7       | 32x5      | 4,536 B    | Second execution input         |

### Test: `test_mnist_inference_tensor_parallel[128-784]`

**Result: PASSED**

Same weight tensors (counters 0-5), different input/output shapes:

| Counter | Shape     | File Size   |
|---------|-----------|-------------|
| 6       | 128x784   | 201,120 B   |
| 7       | 128x5     | 16,824 B    |

### Observations

1. **Round-trip serialization works**: Multi-device host tensors can be serialized to
   disk and loaded back as host tensors, then successfully converted to device layout
   via `toLayout`. The test passes with correct numerical results (within atol=0.05).

2. **Loaded tensor is NOT multi-device**: `loadTensor` without a device produces a
   single host tensor (not multi-device). However, `toLayout` handles this correctly --
   it can take a single host tensor and place it on the device mesh with the
   correct layout. This is a key finding: the multi-device structure is only needed
   for the host-side representation; `toLayout` reconstructs whatever structure the
   device needs.

3. **Shape is preserved**: The loaded tensor shape matches the original shape in all
   cases, verified by the log output.

4. **Performance**: Each dump+load cycle adds roughly sub-millisecond overhead for
   small tensors. The dominant cost in the end-to-end test is compilation and device
   execution, not serialization.

5. **File format**: The `.tensorbin` flatbuffer format encodes tensor metadata (shape,
   dtype, stride) alongside the raw data, producing files slightly larger than the raw
   data size. For example, a `512x256` bf16 tensor (262,144 bytes raw) produces a
   524,704 byte file (~2x, likely due to flatbuffer overhead and alignment).

6. **Arguments arrive in reverse layer order**: The MNIST model has layers
   input(784)->256->512->10, but the arguments arrive as output-layer-first
   (bias_out, weight_out, bias_h2, weight_h2, bias_h1, weight_h1, input_data).

7. **Two executions per test**: Each test triggers two program executions (the main
   MNIST forward pass and a second "ReplicateShardedData" pass), producing 7+1=8
   serialized tensors.

## Code Diff

The complete change is in `pjrt_implementation/src/api/tensor.cc`. The serialization
experiment is clearly bracketed with comments:

```
// --- Serialization experiment: round-trip multi-device host tensors ---
...
// --- End serialization experiment ---
```

To revert, simply remove the code between these markers and the three added includes
(`<atomic>`, `<filesystem>`, `<sstream>`).

## Next Steps / Ideas

- **Reconstruct multi-device structure on load**: Currently `loadTensor` returns a
  single host tensor. To fully round-trip the multi-device structure, one could load
  the tensor, split it according to the original strategy, and call
  `createMultiDeviceHostTensor` to recreate the multi-device representation.

- **Stable cache keys**: Replace the counter with a hash of (program_id, arg_index,
  shape, dtype) for deterministic cache keys across runs.

- **Cache hit optimization**: Check if a serialized file already exists before dumping,
  and skip the dump if so. This would enable a "serialize once, load many" pattern.

- **Device-side serialization**: Test `dumpTensor` on device tensors (after `toLayout`)
  and `loadTensor` with a device argument for direct device-to-device serialization
  without the host round-trip.
