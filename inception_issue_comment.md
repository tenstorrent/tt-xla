## Root Cause Analysis — Inception v4 PCC Drop / AvgPool2d Compiler Bug

**Failing test:** `inception/pytorch-v4-single_device-inference`
**Observed PCC:** 0.014 (garbage output) | **Required:** 0.96
**Repro branch:** `lelanchelian/inception-v4-avgpool-repro`

---

### Responsible Op & Exact Location

**Op:** `torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)`

**Location in model:**
```
timm inception_v4
└── features[6]  InceptionA
     └── branch3[0]  AvgPool2d(kernel_size=3, stride=1, padding=1)
```

This same `AvgPool2d(k=3, s=1, p=1)` pattern appears in **every** inception block:
- `InceptionA` × 4 (features[6–9])
- `InceptionB` × 7 (features[11–17])
- `InceptionC` × 3 (features[19–21])

TT produces `inf` at the first occurrence (features[6].branch3), which then
propagates as `nan`/`inf` through all subsequent blocks → PCC = 0.014 at model output.

---

### What's Going On

Inception v4 outputs near-random values on TT hardware. After slicing through the model
block by block, we isolated the exact operation responsible: `AvgPool2d(kernel_size=3,
stride=1, padding=1)` inside `InceptionA.branch3`. Everything upstream is clean and
numerically correct — the `inf` is born at that single op and then floods every layer
that follows.

---

### Bug Summary

`nn.AvgPool2d` with **kernel_size > 1** produces **NaN/Inf** on TT hardware.
This is the **same compiler bug** already documented for DenseNet121.

| Op | kernel_size | stride | padding | TT result |
|---|---|---|---|---|
| `AvgPool2d` | 3 | 1 | 1 | **Inf** ❌ |
| `AvgPool2d` | 2 | 2 | 0 | **NaN** ❌ (DenseNet121) |
| `MaxPool2d` | 3 | 2 | 0 | Correct ✅ |

---

### Inception v4 — Model Structure

```
Inception v4 forward path (timm)
  features[0-2]   ConvNormAct × 3  (stem)            PASS ✅  PCC=1.000
  features[3]     Mixed3a  (MaxPool2d + Conv3×3)      PASS ✅  PCC=0.9999
  features[4]     Mixed4a                             PASS ✅
  features[5]     Mixed5a                             PASS ✅
  features[6]     InceptionA
    ├─ branch0    ConvNormAct(384→96, 1×1)            PASS ✅  PCC=1.000
    ├─ branch1    ConvNormAct(384→64) → Conv(64→96)   PASS ✅  PCC=0.9999
    ├─ branch2    ConvNormAct × 3                     PASS ✅  PCC=0.9999
    └─ branch3
         ├─ [0]  AvgPool2d(k=3, s=1, p=1)  ← Inf born here ❌
         └─ [1]  ConvNormAct(384→96, 1×1)   downstream Inf ❌
  features[7-9]   InceptionA × 3    all Inf (propagated) ❌
  features[10]    ReductionA                          Inf ❌
  features[11-17] InceptionB × 7                     Inf ❌
  features[18]    ReductionB                          Inf ❌
  features[19-21] InceptionC × 3                     Inf ❌
  global_pool     SelectAdaptivePool2d                Inf ❌
  last_linear     Linear(1536→1000)    PCC = 0.014   ❌
```

Key sizes at the failure point:
- Input to `InceptionA.branch3[0]`:  `[1, 384, 35, 35]` — **clean, PCC≈1.0**
- Output of `InceptionA.branch3[0]`: `[1, 384, 35, 35]` — **Inf on TT**

---

### Exact Failure Statistics

**Input to AvgPool2d — CLEAN:**
```
CPU: shape=[1,384,35,35]  min=0.0000  max=12.5068  mean=0.738  std=1.207  nan=False  inf=False
TT:  shape=[1,384,35,35]  min=0.0000  max=12.5068  mean=0.738  std=1.207  nan=False  inf=False
PCC ≈ 1.0
```

**Output of AvgPool2d — BROKEN:**
```
CPU: shape=[1,384,35,35]  min=0.0632  max=2.2431  mean=0.930  std=0.260  nan=False  inf=False
TT:  shape=[1,384,35,35]  min=0.0000  max=inf      mean=inf    std=nan   nan=False  inf=True
PCC = nan
```

---

### Minimal Standalone Repro

No model download needed:

```python
import torch
import torch_xla
import torch_plugin_tt  # noqa

device = torch_xla.device()
x = torch.abs(torch.randn(1, 384, 35, 35))
pool = torch.nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

cpu_out = pool(x)
tt_out = torch.compile(pool.to(device), backend="tt")(x.to(device)).cpu()

print(cpu_out.isinf().any())  # False
print(tt_out.isinf().any())   # True  ← BUG
```

---

### Repro Steps

```bash
git checkout lelanchelian/inception-v4-avgpool-repro

# Minimal standalone (no model download, < 2 min):
pytest -svv tests/torch/graphs/test_inception_avgpool_nan.py::test_inception_v4_inceptiona_branch3_avgpool_nan

# Full model test (~15 min):
pytest -svv "tests/runner/test_models.py::test_all_models_torch[inception/pytorch-v4-single_device-inference]"
```

---

### Debug Logs

All logs are in `inception/` on the repro branch:
- `inception/run_output.log` — Stage 1 coarse sweep
- `inception/stage2_output.log` — InceptionA block drill-down
- `inception/stage3_output.log` — branch-level isolation
- `inception/sanity_avgpool.log` — standalone AvgPool2d confirmation
- `inception_pcc_debug.py` — full progressive slicer script
- `inception_stage2.py` / `inception_stage3.py` — drill-down scripts
- `inception_sanity_avgpool.py` — minimal standalone repro script
- `tests/torch/graphs/test_inception_avgpool_nan.py` — pytest sanity test

---

### Relationship to DenseNet121 Issue

This is the **same underlying compiler bug** as the DenseNet121 issue:

| | DenseNet121 | Inception v4 |
|---|---|---|
| Symptom | NaN (pcc=nan) | Inf → NaN (pcc=0.014) |
| Failing op | `AvgPool2d(k=2, s=2)` | `AvgPool2d(k=3, s=1, p=1)` |
| First failure point | `features.transition1.pool` | `features[6].branch3[0]` |
| Input shape | `[1, 128, 56, 56]` | `[1, 384, 35, 35]` |
| # affected blocks | transition1–3 (3 ops) | InceptionA/B/C (14 blocks, 14 ops) |
| Root cause | TT-MLIR AvgPool2d lowering bug | same |

Fixing the `AvgPool2d` compiler bug will unblock both DenseNet121 and Inception v4.
