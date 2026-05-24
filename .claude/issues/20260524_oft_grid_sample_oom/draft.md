### Describe the bug

- `oft/pytorch-single_device-inference` (pytorch_OFT_base_cv_object_det_custom) — DRAM OOM on n150 in the lowering of `torch.nn.functional.grid_sample`.
- Surface error from PJRT/XLA can be opaque (`Bad StatusOr access: INTERNAL: Error code: 13`); underlying TT-Metal error is `TT_FATAL ... Out of Memory ... DRAM buffer`.
- Same op/path as the earlier grid_sample OOM in #3419, but the offending allocation is now ~4× smaller on `main` (i.e. partial improvement landed, root cause still present).

#### Call chain (where it shows up)
```
OFT (object-detection model)
  → backbone / feature map (1, 256, 28, 28)
      → torch.nn.functional.grid_sample(input, grid)        # grid: (1, 7, 25281, 2)
          → StableHLO grid_sample lowering
              → ttnn.repeat (RepeatDeviceOperation::create_output_tensors / repeat_upper_dims_rm)
              → ttnn.add   (BinaryNgDeviceOperation::create_output_tensors)
```

#### Key observations

- Isolated `grid_sample` sanity with shapes `input=(1, 256, 28, 28)`, `grid=(1, 7, 25281, 2)` hits OOM on `ttnn.repeat` requesting ~1,449,713,664 B DRAM. This is the dominant allocation in the grid_sample lowering.
- The model-level run in `/tmp/oft4244_model.log` proximally OOMs on `ttnn.add` (`BinaryNgDeviceOperation::create_output_tensors`) requesting 838,860,800 B DRAM across 12 banks (each bank needs 69,906,432 B; free per-bank only ~29 MB at that point) — i.e. allocator is already saturated by upstream tensors produced inside the same grid_sample lowering.
- Same lowering path as #3419. Allocation in the sanity is ~4× smaller than what was reported there, so a previous fix reduced the blast but didn't fully eliminate the materialization. The fundamental issue (materializing a fully-broadcast/repeated intermediate before the add) is unchanged.
- This is a lowering/op-implementation issue, not a model-config issue: shapes are within what other paths consume comfortably on n150.

#### Experiments / sanities

| Test | Result | Notes |
|------|--------|-------|
| Whole model `oft/pytorch-single_device-inference` | OOM (`ttnn.add`, 838,860,800 B DRAM) | See `/tmp/oft4244_model.log` line 29 onward |
| Isolated `grid_sample` sanity (input `(1,256,28,28)`, grid `(1,7,25281,2)`) | OOM (`ttnn.repeat`, ~1,449,713,664 B DRAM) | Same lowering path as #3419, ~4× smaller alloc than #3419 |
| #3419 reference | OOM at `ttnn.repeat` (larger alloc) | Same `RepeatDeviceOperation` / `repeat_upper_dims_rm` |

### Steps to reproduce the issue

```bash
# Whole-model repro (current main)
pytest -svv "tests/runner/test_models.py::test_all_models_torch[oft/pytorch-single_device-inference]"
```

Isolated `grid_sample` sanity: same input/grid shapes as above driven through `torch.nn.functional.grid_sample` under `tt` backend; failure reproduces on bare op call (see #3419 for sanity scaffolding pattern).

### Logs

- `/tmp/oft4244_model.log` — whole-model failure; contains the TT_FATAL OOM stack at `ttnn.add` (line 29 onward, full backtrace through `tt::pjrt::FlatbufferLoadedExecutableInstance::execute`).
- Sanity OOM at `ttnn.repeat` — see investigation summary above; full trace path is `RepeatDeviceOperation::create_output_tensors` → `repeat_upper_dims_rm`.

Excerpt (model log, line 29):

```
TT_FATAL: Out of Memory: Not enough space to allocate 838860800 B DRAM buffer
across 12 banks, where each bank needs to store 69906432 B, but bank size is
1071821792 B (allocated: 1042764352 B, free: 29057440 B, largest free block:
19864480 B)
... ttnn::operations::binary_ng::BinaryNgDeviceOperation::create_output_tensors ...
```

### Expected behavior

`oft/pytorch-single_device-inference` should compile and execute on n150 without DRAM OOM. The `grid_sample` lowering should not materialize a multi-hundred-MB / >1 GB intermediate for these (very modest) input/grid shapes.

### Related issues

- #3419 — earlier `grid_sample` OOM at `ttnn.repeat`; same op/lowering path, larger allocation. Likely the right place to track the upstream fix; this issue tracks the model-level surface and the still-failing OFT bringup.

### Notes

- Arch: n150
- File classification: tt-xla model issue (lowering/runtime impact surfacing inside a tt-xla model test); op-level root cause may warrant a separate tt-metal issue tied to `RepeatDeviceOperation` if a follow-up unit repro is produced.
