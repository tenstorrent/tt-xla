---
name: model-bringup
description: Step-by-step guide for bringing up a new PyTorch model in the tt-xla tt-forge model test suite. Use when the user wants to add, debug, or move a model to EXPECTED_PASSING status.
---

# Model Bringup for tt-xla

Follow these steps to bring up a new model in `third_party/tt_forge_models/` and get it to `EXPECTED_PASSING` status on TT hardware.

---

## Step 1 — Understand the real model

Before writing any code, read the actual model source:

- What are the inputs and their shapes?
- What ops does it use? Check for XLA-incompatible ops:
  - **Dynamic indexing / scatter** (e.g., `tensor[:, indices] = values`) — not traceable
  - **`MaxPool3d` with bfloat16** — no tt-mlir support; replace with stride-2 `Conv3d`
  - **`ConvTranspose3d` (lhs_dilation)** — broken in tt-mlir; replace with `F.interpolate + Conv3d(1×1×1)`
  - **Dynamic token shapes** — blocked by "Shardy propagation only supports ranked tensors with a static shape"
  - **Custom CUDA ops** (spconv, DCN, NMS) — must stay on CPU or be replaced
- What parts can run on TT vs what must stay on CPU?

---

## Step 2 — Create the tt-forge model files

Model lives in `third_party/tt_forge_models/<model_name>/pytorch/`:

```
<model_name>/
  __init__.py
  pytorch/
    __init__.py
    loader.py        ← ModelLoader + ModelVariant + load_model() + load_inputs()
    src/
      model.py       ← nn.Module implementation
```

### Rules for `model.py`
- Use only standard PyTorch ops that StableHLO can represent
- All input shapes must be **static** (no dynamic sizes)
- No preprocessing (voxelization, NMS, tokenization) — those stay on CPU
- If the real model has XLA-incompatible ops, replace them with static equivalents and document why in a comment

### Rules for `loader.py`
- `load_model()` must return a model in `eval()` mode
- `load_inputs()` must return a tuple of tensors with fixed shapes
- Use `dtype_override` parameter to support bfloat16 loading
- `ModelInfo.group` should be `ModelGroup.RED` for new models

---

## Step 3 — Add to test config YAML

Edit `tests/runner/test_config/torch/test_config_inference_single_device.yaml`:

```yaml
  <model_name>/pytorch-<VariantName>-single_device-inference:
    status: KNOWN_FAILURE_XFAIL
    bringup_status: IN_PROGRESS
    reason: "Initial bringup"
```

Start with `KNOWN_FAILURE_XFAIL` — only move to `EXPECTED_PASSING` after verifying PCC.

---

## Step 4 — Run the pytest test

```bash
source venv/activate
pytest -svv "tests/runner/test_models.py::test_all_models_torch[<model_name>/pytorch-<VariantName>-single_device-inference]"
```

Common failure modes and fixes:

| Error | Cause | Fix |
|---|---|---|
| `Bad StatusOr access: INTERNAL: Error code: 13` | Compilation error — check tt-mlir logs | Simplify the failing op |
| `Input type (float) and bias type (BFloat16)` | Input dtype mismatch on TT | Cast inputs to bfloat16 before TT device |
| `Statically allocated circular buffers grow to X B, exceeding max L1` | L1 OOM | Reduce channels / batch size |
| `DRAM Auto slice could not find valid slice configuration` | Large dilated conv, can't slice | Avoid large dilation or reduce spatial dims |
| `Shardy propagation only supports ranked tensors with a static shape` | Dynamic shape | Fix input shapes to be fully static |

---

## Step 5 — Verify CPU vs TT outputs (PCC)

Add the model to `scripts/verify_model_cpu_vs_tt.py` MODEL_REGISTRY:

```python
"<model_name>": [
    ModelSpec("<model_name>/<VariantName>",
              "tt_forge_models.<model_name>.pytorch.loader",
              "<VariantName>"),
],
```

Then run:

```bash
python scripts/verify_model_cpu_vs_tt.py --models <model_name> --verbose
```

**Acceptance criteria:**
- PCC ≥ 0.99 on all output tensors
- No shape mismatches
- CPU reference must use bfloat16 (TT always runs bfloat16; comparing float32 vs bfloat16 inflates errors)

---

## Step 6 — Update test config to EXPECTED_PASSING

Once PCC ≥ 0.99:

```yaml
  <model_name>/pytorch-<VariantName>-single_device-inference:
    status: EXPECTED_PASSING
```

Remove `bringup_status` and `reason` fields.

---

## Known tt-mlir limitations (as of 2026-03)

| Op | Issue | Workaround |
|---|---|---|
| `MaxPool3d` | No bfloat16 support on XLA backends | Replace with stride-2 `Conv3d` |
| `ConvTranspose3d` (lhs_dilation) | Incorrect result in tt-mlir conv3d op | `F.interpolate` + `Conv3d(1×1×1)` |
| `Conv2d` with `dilation=18`, large channels | DRAM slice fatal (output width=44, 512→512ch) | Needs compiler fix |
| `Conv3d` with large `out_channels` (e.g. 1280) | L1 static CB overflow | Needs compiler fix |
| Dynamic token shapes | Shardy only supports static shapes | Not supported for LLMs |
| Large models (>single chip DRAM) | Out of memory | Multi-chip or model splitting |

---

## Quick reference: file locations

| What | Where |
|---|---|
| Model source | `third_party/tt_forge_models/<model>/pytorch/src/model.py` |
| Model loader | `third_party/tt_forge_models/<model>/pytorch/loader.py` |
| Test config | `tests/runner/test_config/torch/test_config_inference_single_device.yaml` |
| CPU vs TT verify script | `scripts/verify_model_cpu_vs_tt.py` |
| Run all models test | `pytest tests/runner/test_models.py -m model_test` |
