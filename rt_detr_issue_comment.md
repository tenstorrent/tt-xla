## Root Cause Analysis — RT-DETR R18vd PCC Drop / AvgPool2d Compiler Bug

**Failing test:** `rt_detr/pytorch-R18vd-single_device-inference`
**Observed PCC:** -0.040 (garbage output) | **Required:** 0.99
**xfail reason (stale):** "Out of Memory" — model actually runs fine, real failure is PCC
**Repro branch:** `lelanchelian/rt-detr-avgpool-repro`

---

### Responsible Op & Exact Location

**Op:** `torch.nn.AvgPool2d(kernel_size=2, stride=2, padding=0)`

**Location in model:**
```
PekingU/rtdetr_r18vd
└── model.backbone.model.encoder.stages[1]
     └── layers[0].shortcut[0]  AvgPool2d(kernel_size=2, stride=2, padding=0)
```

The same `AvgPool2d(k=2, s=2)` shortcut appears in **stages[1], [2], and [3]** —
the first occurrence (stage[1]) corrupts activations with `inf`/`nan`, which then
propagates through all downstream stages.

---

### What's Going On

RT-DETR R18vd outputs near-random values on TT hardware. The xfail reason ("Out of Memory")
is stale — the model runs successfully, but PCC = -0.040. After slicing through the
ResNet backbone stage by stage, the exact failure point is:

`AvgPool2d(kernel_size=2, stride=2, padding=0)` in the shortcut path of
`RTDetrResNetBasicLayer` at `stages[1].layers[0]`.

This is the **exact same compiler bug** as DenseNet121 — `AvgPool2d(k=2, s=2)`.

---

### Bug Summary

`nn.AvgPool2d` with **kernel_size > 1** produces **NaN/Inf** on TT hardware.
Third model confirmed with this bug.

| Op | kernel_size | stride | padding | TT result | Model |
|---|---|---|---|---|---|
| `AvgPool2d` | 2 | 2 | 0 | **NaN** ❌ | DenseNet121 |
| `AvgPool2d` | 2 | 2 | 0 | **NaN/Inf** ❌ | RT-DETR R18vd |
| `AvgPool2d` | 3 | 1 | 1 | **Inf** ❌ | Inception v4 |
| `MaxPool2d` | 3 | 2 | 1 | Correct ✅ | — |

---

### RT-DETR R18vd — Model Structure

```
RT-DETR R18vd backbone (RTDetrResNetBackbone)
  embedder     Conv×3 (3→32→32→64) + MaxPool2d(k=3,s=2)    PASS ✅  PCC=0.9999
  stage[0]     RTDetrResNetStage  (64→64,  no downsampling)  PASS ✅  PCC=1.0001
  stage[1]     RTDetrResNetStage  (64→128)
    └── layer[0]  RTDetrResNetBasicLayer
          ├── shortcut
          │    ├── [0] AvgPool2d(k=2,s=2)  ← NaN/Inf born here ❌
          │    └── [1] RTDetrResNetShortCut (Conv2d 64→128, 1×1)
          └── layer  Conv+BN+ReLU × 2
  stage[2]     RTDetrResNetStage  (128→256)  — Inf propagated ❌
  stage[3]     RTDetrResNetStage  (256→512)  — Inf propagated ❌
  → encoder (AIFI), decoder, heads  — all garbage ❌
  Final PCC = -0.040
```

Key sizes at the failure point:
- Input to `AvgPool2d`:  `[1, 64, 160, 160]` — **clean, PCC≈1.0**
- Output of `AvgPool2d`: `[1, 64, 80, 80]`  — **NaN and Inf on TT**

---

### Exact Failure Statistics

**Input to AvgPool2d — CLEAN:**
```
CPU: shape=[1,64,160,160]  min=0.0000  max=9.4336  mean=0.386  std=0.441  nan=False  inf=False
TT:  shape=[1,64,160,160]  min=0.0000  max=9.4336  mean=0.386  std=0.441  nan=False  inf=False
PCC = 1.000144
```

**Output of AvgPool2d — BROKEN:**
```
CPU: shape=[1,64,80,80]  min=0.0000  max=3.6891  mean=0.386  std=0.376  nan=False  inf=False
TT:  shape=[1,64,80,80]  min=nan     max=nan      mean=nan    std=nan   nan=True   inf=True
PCC = nan
```

---

### Note on Stale xfail Reason

The current xfail reason says "Out of Memory: Not enough space to allocate 314572800 B DRAM buffer."
This is **outdated** — the model runs successfully (completes in ~4 min) and the actual
failure is a PCC drop caused by the AvgPool2d bug. The xfail reason should be updated to
reflect the real root cause.

---

### Repro Steps

```bash
git checkout lelanchelian/rt-detr-avgpool-repro

# Minimal standalone (no model download, < 1 min):
pytest -svv tests/torch/graphs/test_rt_detr_avgpool_nan.py::test_rt_detr_r18vd_backbone_shortcut_avgpool_nan

# Full model test (~4 min):
pytest -svv "tests/runner/test_models.py::test_all_models_torch[rt_detr/pytorch-R18vd-single_device-inference]"
```

---

### Debug Logs

All logs are in `rt_detr/` on the repro branch:
- `rt_detr/run_output.log` — full progressive sweep
- `rt_detr/pcc_summary.json` — machine-readable results
- `rt_detr_pcc_debug.py` — debug script
- `tests/torch/graphs/test_rt_detr_avgpool_nan.py` — pytest sanity test

---

### Relationship to DenseNet121 and Inception v4

This is the **same underlying compiler bug** across all three models:

| | DenseNet121 | Inception v4 | RT-DETR R18vd |
|---|---|---|---|
| Symptom | NaN (pcc=nan) | Inf→NaN (pcc=0.014) | NaN/Inf (pcc=-0.040) |
| Failing op | `AvgPool2d(k=2,s=2)` | `AvgPool2d(k=3,s=1,p=1)` | `AvgPool2d(k=2,s=2)` |
| Location | transition layer | InceptionA/B/C branch3 | ResNet shortcut path |
| Input shape | `[1,128,56,56]` | `[1,384,35,35]` | `[1,64,160,160]` |
| Root cause | TT-MLIR AvgPool2d lowering bug | same | same |

Fixing the `AvgPool2d` compiler bug will unblock all three models.
