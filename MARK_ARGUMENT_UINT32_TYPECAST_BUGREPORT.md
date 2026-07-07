# `tt.mark_argument` drops uint32 → int32, breaking the StableHLO pipeline

## TL;DR

`tt.mark_argument` is an **identity** annotation op, but for a `uint32` input it is
emitted with a `ui32` operand and an `i32` result. The frontend pass
`PopulateArgumentAttrsFromTTMark` folds the op away by replacing all uses of its
(`i32`) result with its (`ui32`) operand **without any type reconciliation**. Every
consumer that was typed against the `i32` result now receives a `ui32` value, so the
first shape-only consumer fails type verification:

```
loc("reshape.59"): error: 'stablehlo.reshape' op requires compatible element types for all operands and results
... ERR| Failed to run stablehlo pipeline
E   ValueError: Error code: 13
```

Net effect: **any graph whose input is `uint32` and is marked as an argument fails to
compile.** It surfaced on GPT-OSS-120B MoE-fused decode, where the decode `input_ids`
placeholder is `uint32` (to match the device `argmax` output dtype), but it is a
general `uint32`-input problem, not specific to that model.

## Affected configuration

- Backend: `torch.compile(backend="tt")` → PJRT → tt-mlir StableHLO pipeline.
- Trigger: a **`uint32`** tensor that is a **function argument** (so the tt backend
  wraps it in `torch.ops.tt.mark_argument_attributes`, see
  `python_package/tt_torch/backend/passes.py:259`) and is then consumed by any
  shape/elementwise op that verifies operand/result element-type equality
  (`stablehlo.reshape`, `stablehlo.broadcast_in_dim`, etc.).
- First observed: `tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_moe_fused_galaxy`
  on a 4x8 Blackhole galaxy mesh, no-trace decode path, after making the decode
  `input_ids` placeholder `uint32` so it matches the on-device `argmax` output
  (avoids a separate ROW_MAJOR device→device typecast fault at runtime).

## Symptom

Compilation aborts in the frontend StableHLO pipeline and PJRT reports the opaque
`ValueError: Error code: 13`. The real diagnostic is one line above in the log:

```
loc("reshape.59"): error: 'stablehlo.reshape' op requires compatible element types for all operands and results
2026-... module_builder.cc:783  ERR| Failed to run stablehlo pipeline
```

## Root cause

### 1. `tt.mark_argument` is emitted with mismatched operand/result element types

The custom op preserves dtype at the Python level — `mark_argument_attributes`
(`python_package/tt_torch/custom_ops.py:46`) passes `[tensor.dtype]` as the
custom-call result dtype. But when the input is `torch.uint32`, torch-xla lowers the
`stablehlo.custom_call` with a **`ui32` operand and a signless `i32` result**. From the
decode-graph StableHLO dump (`modules/irs/shlo_..._g1_*.mlir`):

```mlir
%arg6: tensor<32x1xui32>                                         // input_ids placeholder (uint32 — correct)
%16 = stablehlo.reshape %arg6 : (tensor<32x1xui32>) -> tensor<1x32x1xui32>          // reshape.56  (ui32 -> ui32, ok)
%17 = stablehlo.custom_call @tt.mark_argument(%16)
        : (tensor<1x32x1xui32>) -> tensor<1x32x1xi32>                                // <-- ui32 IN, i32 OUT
%18 = stablehlo.reshape %17 : (tensor<1x32x1xi32>) -> tensor<32xi32>               // reshape.59
%19 = "stablehlo.gather"(%15, %18) ... : (tensor<201088x2880xbf16>, tensor<32xi32>) -> tensor<32x2880xbf16>
```

At this stage `reshape.59` (`%18`) is internally consistent (`i32 -> i32`) because it
reads the `i32` result of the mark op. The bug is latent.

### 2. Folding the mark op forwards a `ui32` value into `i32`-typed consumers

`PopulateArgumentAttrsFromTTMark`
(`pjrt_implementation/src/api/module_builder/frontend_passes/shlo_input_role_propagation.cc:198`)
lifts the annotation onto the function argument and then erases the op:

```cpp
// shlo_input_role_propagation.cc:286
rewriter.replaceOp(op, input);   // input = %16 (ui32);  op result was i32
```

`replaceOp` rewires every use of the `i32` result to the `ui32` operand with **no cast
inserted and no type check**. `reshape.59` now has a `ui32` operand but an `i32` result,
which violates `stablehlo.reshape`'s "compatible element types" invariant, and the
pipeline fails verification.

### Why it doesn't bite `int32`/`bfloat16`

For those dtypes the mark op's operand and result types are identical, so
`replaceOp(op, input)` is genuinely type-preserving and folding is safe. Only the
signed/unsigned integer round-trip (`ui32` operand vs signless `i32` result) creates the
mismatch.

## Repro

### Authoritative repro (validated)

Make the decode `input_ids` placeholder `uint32` and run the MoE-fused decode test. In
`tests/benchmark/llm_utils/decode_utils.py`, in `LLMSamplingWrapper.forward`, cast the
per-step token ids to `uint32` on device:

```python
next_token_ids = logits[:, -1].argmax(dim=-1, keepdim=True)
if next_token_ids.device.type == "xla":
    next_token_ids = next_token_ids.to(torch.uint32)   # placeholder becomes ui32
```

Then:

```bash
pytest -svv --num-layers 2 \
  tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_moe_fused_galaxy
```

Observed (this is the failure this doc is about): the decode graph (graph 1) compiles its
`input_ids` placeholder as `tensor<32x1xui32>`, then the frontend StableHLO pipeline
aborts:

```
loc("reshape.59"): error: 'stablehlo.reshape' op requires compatible element types for all operands and results
module_builder.cc:783  ERR| Failed to run stablehlo pipeline
E   ValueError: Error code: 13
```

The exact offending op sequence is the IR excerpt shown under **Root cause** above,
extracted from the dumped `modules/irs/shlo_..._g1_*.mlir` for this run.

### Standalone kernel sketch (IR-derived, not yet validated on-device)

The essence is a `uint32` function argument (which the tt backend wraps in
`tt.mark_argument_attributes`, `python_package/tt_torch/backend/passes.py:259`) feeding a
shape-only consumer. The following isolates that pattern; it has **not** been confirmed
to reproduce standalone (an attempt was blocked by an unrelated galaxy fabric-init issue
on a bare, mesh-less run), so treat the test above as the source of truth until this is
validated:

```python
import torch, torch_xla
import torch_xla.runtime as xr
xr.set_device_type("TT")
import tt_torch  # registers the "tt" backend and torch.ops.tt.* custom ops

dev = torch_xla.device()

def f(x):
    x = torch.ops.tt.mark_argument_attributes(x, "input", "input_ids")
    return x.reshape(-1) + 1

# uint32 -> ui32 operand / i32 result on the mark op; int32 compiles cleanly.
torch.compile(f, backend="tt")(torch.zeros(8, 1, dtype=torch.uint32, device=dev)).cpu()
```

## Intended solution (A): make `tt.mark_argument` truly type-preserving for uint32

`tt.mark_argument` must be an identity in **type** as well as value. Preferred fix, in
order of where the invariant is cheapest to guarantee:

1. **Emit a matching result type (root fix).** Ensure the `stablehlo.custom_call` for
   `tt.mark_argument` on a `uint32` value has a `ui32` result, not `i32`. If torch-xla's
   `stablehlo_custom_call` coerces `uint32` result types to signless `i32` while leaving
   the operand `ui32`, fix that lowering (or choose a result-type spelling that
   round-trips) so operand and result agree.

2. **Guard the fold (defensive fix).** In `PopulateArgumentAttrsFromTTMark`
   (`shlo_input_role_propagation.cc:286`), before `replaceOp(op, input)`, assert
   `op.getResult(0).getType() == input.getType()`; if they differ, insert a
   `stablehlo.convert`/`bitcast_convert` from the operand type to the result type
   instead of a raw replace. This keeps the pass from silently producing ill-typed IR
   and turns any future dtype round-trip gap into an explicit, localized convert.

3. **Verify downstream `uint32` support.** Once the value stays `ui32`, confirm the
   consumers on the `input_ids` path accept unsigned indices — notably
   `stablehlo.gather` (embedding lookup) and the tt-mlir → TTNN lowering of gather /
   `embedding` with `ui32` index tensors. If TTNN requires signed indices, the correct
   place for the (single, explicit) `ui32 → i32` convert is at the gather boundary, not
   smuggled through an identity annotation op.

The goal state: `input_ids` is genuinely `uint32` end-to-end (which is the natural dtype
for non-negative token ids), the mark op is a true no-op, and no implicit element-type
change is introduced by folding.

## References

- `python_package/tt_torch/custom_ops.py:14` — `tt::mark_argument_attributes` op
  definition (declares result dtype `[tensor.dtype]`).
- `python_package/tt_torch/backend/passes.py:259` — where graph inputs are wrapped in
  `mark_argument_attributes`.
- `pjrt_implementation/src/api/module_builder/frontend_passes/shlo_input_role_propagation.cc:198`
  — `PopulateArgumentAttrsFromTTMark`; the offending fold is the `replaceOp(op, input)`
  at line 286.
- tt-mlir: `runtime/lib/ttnn/types/layout_converter.cpp` — the *separate*, downstream
  ROW_MAJOR device→device typecast limitation that motivated making the placeholder
  `uint32` in the first place.
- Example failing test:
  `tests/benchmark/test_llms.py::test_gpt_oss_120b_tp_moe_fused_galaxy`.
