# Fix: chisel softmax golden ignores the `dimension` attribute

**Symptom:** every `ttnn.softmax` with `dimension != 1` reports a spurious chisel numerics failure
(PCC ≈ 0). The device kernel is correct — proven by a tt-mlir silicon repro (PCC 0.9999).

**Cause:** `chisel_ttnn_softmax` passes the axis as `dimension=`, but `softmax_golden` reads the key
`"dim"` and so always defaults to axis 1.

## The fix (one line)

File: `tools/golden/mapping.py` (in tt-mlir), function `softmax_golden` (~line 3053).

```diff
 def softmax_golden(input_tensor: GoldenMapTensor, **kwargs) -> GoldenMapTensor:
-    dimension = kwargs.get("dim", 1)
+    dimension = kwargs.get("dimension", kwargs.get("dim", 1))
     return torch.nn.functional.softmax(input_tensor, dim=dimension)
```

Reads the key the caller actually sends (`dimension`), keeping `dim` as a fallback for any other
caller.

## Verify (seconds, no device / model)

In the tt-mlir checkout (with the venv that has the `golden` package), run **before** and **after**
the edit:

```bash
python -c "import torch; from golden.mapping import softmax_golden as s; x=torch.randn(2,4,8,16); print('honors dimension=3:', torch.allclose(s(x, dimension=3), torch.softmax(x, dim=3)))"
```

* **Before fix:** `honors dimension=3: False`  (golden softmaxes the wrong axis → the bug)
* **After fix:**  `honors dimension=3: True`   (fixed)

Full proof test (3 assertions, ~1s): `chisel_analysis/repros/test_chisel_softmax_golden_bug.py`.
After applying the fix, its `test_softmax_golden_ignores_dimension_kwarg` (which asserts the *buggy*
behavior) is expected to flip to FAIL — that flip is the confirmation the fix took.
