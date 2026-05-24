### Upstream context
- Reproduced from **tenstorrent/tt-forge-fe** — `grid_sample` op lowering exercising `torch.floor` on a small 1-D tensor.
- Original branch: `mramanathan/grid_sample_floor_issue`.
- Original repro: `pytest forge/test/mlir/test_ops.py::test_floor`.
- Failure stage: TTMLIR compilation → tt-metal runtime (bcast program factory aborts during compile of the lowered `ttir.floor` decomposition).
- Hardware: n150.

> **Status:** historical — not reproducible on current `main`. Draft retained for template validation.

### Summary
The `ttir.floor` decomposition produced by lowering `torch.floor` (as used inside the `grid_sample` op path) lowers into a small graph containing `ttir.le`, `typecast`, `multiply`, and `subtract`. The `multiply`/`subtract` step is emitted in a form that the broadcast height/width multi-core program factory cannot map, and compilation aborts with `BinaryOpType cannot be mapped to BcastOpMath`.

### Error
```
RuntimeError: TT_THROW @ tt_metal/.../broadcast_height_and_width_multi_core_program_factory.cpp:27
info: BinaryOpType cannot be mapped to BcastOpMath
```

### Key observations
- Input tensor for the failing `floor` is shape `tensor<6xf32>` (1-D, 6 elements).
- Failure is triggered by the `floor` decomposition emitted during `grid_sample` lowering; it is not specific to `grid_sample` itself — any `floor` on this shape exercised the same path.
- The aborting kernel is the broadcast height-and-width multi-core program factory, invoked while lowering the `multiply`/`subtract` step of the decomposition.
- Error surfaces at compile/program-factory time, not at runtime data movement.

### Affected graph / op

Full TTIR graph for the failing `floor` decomposition was captured in the original issue. Schematic:

```
// floor(x) decomposition on tensor<6xf32>
%0 = ttir.le      %x, %x_floor_candidate   : tensor<6xi1>
%1 = ttir.typecast %0                       : tensor<6xf32>
%2 = ttir.multiply %1, %const               : tensor<6xf32>
%3 = ttir.subtract %x_floor_candidate, %2   : tensor<6xf32>
// %3 == floor(%x)
```

The `multiply` / `subtract` ops are what hit the bcast program factory and fail the `BinaryOpType -> BcastOpMath` mapping.

### Steps to reproduce
```bash
# tt-forge-fe side (historical)
git checkout mramanathan/grid_sample_floor_issue
pytest -svv forge/test/mlir/test_ops.py::test_floor
```

No standalone `mlir-opt` repro was captured at the time.

### Logs
- `floor.log` — historical, **not attached / unavailable**.

### Expected behavior
`ttir.floor` (and the `multiply` / `subtract` it decomposes into) lowers successfully on `tensor<6xf32>` without hitting the `BinaryOpType cannot be mapped to BcastOpMath` path; `grid_sample` compile completes and reaches runtime.
