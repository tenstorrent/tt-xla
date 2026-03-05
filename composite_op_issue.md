# `StableHLOCompositeBuilder`: crash in `_xla_warm_up_cache` when a multi-output composite has dead (unused) marked outputs

## Description

When using `StableHLOCompositeBuilder` to wrap a multi-output operation (e.g. `torch.topk`), XLA crashes during compilation if only a subset of the marked outputs are actually consumed by the rest of the model.

`mark_outputs` assigns each output tensor a positional index (`pos=0`, `pos=1`, …). XLA's composite detection pass expects all marked positions to be reachable in the HLO graph. When an output is unused, XLA's dead-code elimination (DCE) removes the corresponding `mark_output` node, leaving a gap in the position sequence. The composite detection pass then crashes because it cannot build a valid composite with non-contiguous output positions.

## Minimal Reproduction

```python
import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch_xla.experimental.mark_pattern_utils import StableHLOCompositeBuilder

def composite_topk_both(x, k):
    """Works fine — both outputs are consumed."""
    builder = StableHLOCompositeBuilder(
        name="mylib.topk", attr={"k": k}
    )
    x = builder.mark_inputs(x)
    values, indices = torch.topk(x, k)
    values, indices = builder.mark_outputs(values, indices)
    return values, indices  # both used

def composite_topk_indices_only(x, k):
    """Crashes — values output is dead after DCE."""
    builder = StableHLOCompositeBuilder(
        name="mylib.topk", attr={"k": k}
    )
    x = builder.mark_inputs(x)
    values, indices = torch.topk(x, k)
    values, indices = builder.mark_outputs(values, indices)
    return indices  # only indices used; values is dead code

device = xm.xla_device()
x = torch.randn(1, 20, device=device)

# Works:
composite_topk_both(x, 5)
xm.mark_step()

# Crashes in _xla_warm_up_cache:
composite_topk_indices_only(x, 5)
xm.mark_step()
```

## Expected Behavior

Composite detection should gracefully handle the case where some marked outputs become dead code after DCE. Only the live marked outputs should be included in the `stablehlo.composite`, with positions renumbered to be a contiguous `0..N-1` sequence.

## Actual Behavior

XLA crashes (typically a segfault or assertion failure) in `_xla_warm_up_cache` when the composite detection pass encounters a gap in output positions (e.g., `pos=1` exists but `pos=0` was DCE'd).

## Root Cause

`StableHLOCompositeBuilder.mark_outputs` inserts a custom-call marker for each output at a fixed position. If the user (or a downstream pass) does not consume a marked output, DCE removes that marker before the composite detection pass runs. The detection pass then finds an incomplete position map and cannot construct a valid composite result list.

The positions need to be re-indexed after DCE, mapping surviving markers to a contiguous range starting at 0.

## Proposed Fix

In the HLO pass that collects `mark_output` custom calls and assembles them into a `stablehlo.composite`:

1. After collecting all `mark_output` nodes for a composite instance, **filter out any with no live users** (i.e., those removed or made dead by DCE).
2. Sort the survivors by their `pos` attribute.
3. Use the survivors as the composite's results, in order — their relative order is preserved even if some intermediate positions were dropped.

This matches the semantics of `mark_outputs(single_tensor)`, which is already valid and assigns `pos=0` to that tensor — the key invariant is that positions are *relatively ordered*, not that they are *globally dense*.

## Workaround

Until this is fixed, callers must create separate composite builders for each output combination they need:

```python
def composite_topk_indices_only(x, k):
    """Workaround: only mark the output(s) that will actually be used."""
    builder = StableHLOCompositeBuilder(
        name="mylib.topk_indices", attr={"k": k}
    )
    x = builder.mark_inputs(x)
    _, indices = torch.topk(x, k)
    indices = builder.mark_outputs(indices)  # single output at pos=0 — safe
    return indices
```

## Environment

- PyTorch version: 2.7.0
- `torch-xla` version: (upstream / custom build)
- Python: 3.12
