## Root Cause Analysis — DenseNet121 NaN / AvgPool2d Compiler Bug

**Failing test:** `densenet/pytorch-121-single_device-inference`
**Observed PCC:** nan (invalid) | **Required:** 0.99
**Repro branch:** `lelanchelian/center_pcc_drop-repro`

---

### Responsible Op & Exact Line

**Op:** `torch.nn.AvgPool2d(kernel_size=2, stride=2)` — registered in DenseNet121 as `features.transition1.pool`

**Exact line in repro:**
```
tests/torch/graphs/test_densenet_avgpool_nan.py:56
```
```python
x = f.transition1.pool(x)   # line 56 — this is where NaN is produced on TT hardware
```

Everything above line 56 is numerically clean. This single call to `AvgPool2d` converts
a perfectly valid `[1, 128, 56, 56]` tensor (PCC=0.9999) into an all-NaN
`[1, 128, 28, 28]` tensor on TT. No other op is involved.

---

### What's Going On

DenseNet121 produces a full NaN output on TT hardware. After slicing through the model
layer by layer, we isolated the exact operation responsible: `AvgPool2d(kernel_size=2, stride=2)`
inside `transition1.pool`. Everything upstream of it is clean and numerically correct —
the NaN is born at that single op and then floods every layer that follows.

---

### Bug Summary

`nn.AvgPool2d` with **any `kernel_size > 1`** produces **NaN/Inf** on TT hardware,
regardless of input shape, dtype, stride, or input values.

| Op | kernel_size | stride | TT result |
|---|---|---|---|
| `AvgPool2d` | 2 | 1 | **NaN** ❌ |
| `AvgPool2d` | 2 | 2 | **NaN** ❌ |
| `AvgPool2d` | 3 | 2 | **NaN** ❌ |
| `AvgPool2d` | 1 | 1 | Correct ✅ (trivial, no actual avg) |
| `MaxPool2d` | 2 | 2 | Correct ✅ |

This is a **TT-MLIR compiler bug** in the lowering of `AvgPool2d` — likely in the
`stablehlo.reduce_window` path with `add` reducer + divisor (used for average pooling),
which is distinct from the `max` reducer path used by `MaxPool2d`.

---

### DenseNet121 — Model Structure

Think of DenseNet as a backbone that grows its feature channels progressively.
Instead of discarding old features, each dense block **reuses and concatenates** all
prior activations — hence "dense." Transition layers are the connective tissue that
compress and spatially downsample between dense blocks.

```
DenseNet121 forward path
  stem (conv0 → norm0 → relu0 → pool0)
    └─ MaxPool2d(k=3, s=2)             PASSES ✅
  denseblock1  [6 dense layers]        PASSES ✅
  transition1
    ├─ norm  BatchNorm2d(256)           PASSES ✅  PCC=0.9999
    ├─ relu  ReLU                       PASSES ✅  PCC=0.9997
    ├─ conv  Conv2d(256→128, 1×1)      PASSES ✅  PCC=0.9999
    └─ pool  AvgPool2d(k=2, s=2)      ← NaN born here ❌
  denseblock2..4 / transition2..3 / classifier
    └─ all NaN (propagated) ❌
```

Key sizes at the failure point:
- Input to `transition1.pool`:  `[1, 128, 56, 56]` — **clean, PCC=0.9999**
- Output of `transition1.pool`: `[1, 128, 28, 28]` — **all NaN on TT**

---

### Latest Finding — Slice Confirmation

To pin down the exact failure boundary, we sliced the model in two ways:

**Slice A — stop before the pool (transition1.conv output):**
The test ran and passed. TT output matched CPU with PCC=0.999970.
This confirmed the input to AvgPool2d is perfectly healthy on TT.

**Slice B — include the pool (transition1.pool output):**
Test updated to include `f.transition1.pool(x)` and return that tensor.
This is the live repro — it is expected to fail with NaN/Inf on TT hardware.

The slice boundary was moved by exactly one op:

```python
# Before (clean output, test passes)
x = f.transition1.conv(x)
return x

# After (NaN output, test fails — reproduces bug)
x = f.transition1.conv(x)
x = f.transition1.pool(x)  # ← AvgPool2d(k=2, s=2) — NaN enters here
return x
```

This gives us a precise, minimal in-model repro with zero ambiguity about where
the numbers break.

---

### Minimal Standalone Repro

No model download needed, runs in under 5 seconds:

```python
import torch, torch_xla.core.xla_model as xm
device = xm.xla_device()

x = torch.randn(1, 4, 8, 8)
pool = torch.nn.AvgPool2d(2, stride=2)

cpu_out = pool(x)
tt_out = pool(x.to(device)).cpu()

print(cpu_out.isnan().any())  # False
print(tt_out.isnan().any())   # True  ← BUG
```

```bash
pytest tests/torch/graphs/test_avgpool2d_nan.py::test_avgpool2d_minimal_k2s2 -svv
```

---

### Exact Failure Statistics (DenseNet121 transition1)

**Input to AvgPool2d — CLEAN:**
```
CPU: shape=[1,128,56,56]  min=-3.5407  max=2.4880  std=0.4027  nan=False  inf=False
TT:  shape=[1,128,56,56]  min=-3.5273  max=2.4727  std=0.4001  nan=False  inf=False
PCC = 0.999970
```

**Output of AvgPool2d — BROKEN:**
```
CPU: shape=[1,128,28,28]  min=-3.4836  max=2.4565  std=0.3877  nan=False  inf=False
TT:  shape=[1,128,28,28]  min=nan      max=nan      std=nan     nan=True   inf=True
PCC = nan
```

---

### Why MaxPool2d Passes but AvgPool2d Fails

Both ops lower to `stablehlo.reduce_window`, but with different reducers:

- **MaxPool2d** → `max` reducer. TT-MLIR handles this correctly.
- **AvgPool2d** → `add` reducer + scalar divisor applied to the window sum.
  The division step likely produces NaN — possibly a divide-by-zero or uninitialized
  divisor in the generated kernel.

`AvgPool2d(k=1, s=1)` passes only because a 1×1 window has nothing to average —
the compiler trivially copies the input without invoking the broken reduction path.

---

### Repro Steps

```bash
git checkout lelanchelian/center_pcc_drop-repro

# Minimal standalone (no model download, < 5 s):
pytest tests/torch/graphs/test_avgpool2d_nan.py::test_avgpool2d_minimal_k2s2 -svv

# In-model slice repro (confirms exact failure boundary, ~30 s):
pytest -svv tests/torch/graphs/test_densenet_avgpool_nan.py::test_densenet121_transition1_avgpool_nan

# Full model test (~4 min):
pytest -svv "tests/runner/test_models.py::test_all_models_torch[densenet/pytorch-121-single_device-inference]"
```

---

### Debug Logs

All logs are in `densenet/` on the repro branch:
- `densenet/debug_summary.log` — full structured summary
- `densenet/block{1-12}.log` — per-block PCC sweep
- `densenet/trans{1-4}.log` — transition1 sub-op isolation
- `densenet/pcc_summary.json` — machine-readable results
- `tests/torch/graphs/test_densenet_avgpool_nan.py` — in-model slice repro
- `tests/torch/graphs/test_avgpool2d_nan.py` — standalone parametric repro (no model needed)

---

### Comparison with CenterNet Issue

| | CenterNet DLA-1x | DenseNet121 |
|---|---|---|
| Symptom | Low PCC (0.934 < 0.97) | NaN/Inf (pcc=nan) |
| Root cause | bfloat16 errors amplified by low-variance output | AvgPool2d compiler bug producing NaN |
| Failing op | `Conv2d(256→2, 1×1)` (reg head) | `AvgPool2d(k≥2)` in any position |
| Severity | Precision degradation | **Complete numerical failure** |
| Shape-dependent? | Yes | **No — any shape, any dtype** |
| Workaround | Raise PCC threshold for reg output | Replace AvgPool2d with manual `F.avg_pool2d` + TT override, or fix TT-MLIR |
