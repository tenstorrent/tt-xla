---
name: model-bringup
description: Step-by-step guide for bringing up a new PyTorch model in the tt-xla tt-forge model test suite. Use when the user wants to add, debug, or move a model to EXPECTED_PASSING status.
---

# Model Bringup for tt-xla

The golden rule: **establish CPU ground truth first, then bring up on TT and verify against it.**

Never bring up a model on TT without first knowing what the correct output looks like on CPU.

---

## Step 1 — Set up local environment and understand the model

Clone or locate the original model repo. Install its dependencies in a separate venv if needed (do not pollute the tt-xla venv).

Read the model source to understand:
- What are the inputs and their shapes? (batch size, spatial dims, channels)
- What preprocessing does it require? (voxelization, tokenization, normalization)
- What is the output? (bounding boxes, heatmaps, logits, embeddings)
- Does it require a pretrained checkpoint, or do random weights suffice for shape correctness?

```bash
# Run the model on CPU with real or synthetic inputs to understand it end-to-end
python run_cpu_demo.py
```

At the end of this step you must know:
- Exact input tensor shapes and dtypes
- Exact output tensor shapes
- What the outputs look like numerically (ranges, distributions)

---

## Step 2 — Run on CPU and save ground truth outputs

Run the model on CPU and save the outputs. These are your ground truth for all TT comparisons.

Cast model and inputs to **bfloat16** before saving — TT hardware always runs in bfloat16, so the ground truth must match that dtype to avoid inflated errors.

```python
import torch

model.eval()
model = model.to(torch.bfloat16)
inputs = [x.to(torch.bfloat16) if x.is_floating_point() else x for x in inputs]

with torch.no_grad():
    cpu_output = model(*inputs)

# Save for later comparison against TT output
torch.save(inputs, "tests/torch/graphs/<model_name>_inputs.pt")
torch.save(cpu_output, "tests/torch/graphs/<model_name>_cpu_output_bf16.pt")
```

Inspect the outputs — understand what "correct" looks like before touching TT.

---

## Step 3 — Identify what can and cannot run on TT (XLA tracing analysis)

Go through the model and flag every op that is not XLA-traceable:


Split the model at the boundary of what TT can run:
- **CPU side:** preprocessing, dynamic ops, NMS / postprocessing
- **TT side:** the neural network backbone / neck / head with static shapes

---

## Step 4 — Create the tt-forge model files

Model lives in `third_party/tt_forge_models/<model_name>/pytorch/`:

```
<model_name>/
  __init__.py
  pytorch/
    __init__.py
    loader.py        ← ModelLoader + ModelVariant + load_model() + load_inputs()
    src/
      model.py       ← nn.Module — only the TT-runnable part
```

### Rules for `model.py`
- Only the neural network portion — no preprocessing, no postprocessing
- All input/output shapes must be static
- If you replaced any ops (MaxPool3d → Conv3d, etc.), add a comment explaining why
- Use only standard PyTorch ops that StableHLO can represent

### Rules for `loader.py`
- `load_inputs()` must return the same shapes measured in Step 1
- Support `dtype_override` for bfloat16
- Return model in `eval()` mode

---

## Step 5 — Add to test config YAML

Edit `tests/runner/test_config/torch/test_config_inference_single_device.yaml`:

```yaml
  <model_name>/pytorch-<VariantName>-single_device-inference:
    status: KNOWN_FAILURE_XFAIL
    bringup_status: IN_PROGRESS
    reason: "Initial bringup"
```

Start with `KNOWN_FAILURE_XFAIL`. Only promote after PCC passes.

---

## Step 6 — Run on TT and verify against CPU ground truth

Run the pytest test:

```bash
source venv/activate
pytest -svv "tests/runner/test_models.py::test_all_models_torch[<model_name>/pytorch-<VariantName>-single_device-inference]"
```

Add the model to `scripts/verify_model_cpu_vs_tt.py` MODEL_REGISTRY and run:

```python
"<model_name>": [
    ModelSpec("<model_name>/<VariantName>",
              "tt_forge_models.<model_name>.pytorch.loader",
              "<VariantName>"),
],
```

```bash
python scripts/verify_model_cpu_vs_tt.py --models <model_name> --verbose
```

The script runs the model on CPU (bfloat16) and on TT device and computes PCC between all output tensors.

**Acceptance criteria — TT output vs CPU bfloat16 ground truth from Step 2:**
- PCC ≥ 0.99 on all output tensors
- No shape mismatches
- Max absolute error consistent with bfloat16 rounding (~5e-4)

Common failures and fixes:

| Error | Cause | Fix |
|---|---|---|
| `Bad StatusOr access: INTERNAL: Error code: 13` | Compilation error | Check tt-mlir logs; simplify the failing op |
| `Input type (float) and bias type (BFloat16)` | Input dtype mismatch on TT | Cast inputs to bfloat16 before sending to TT device |
| `Statically allocated circular buffers grow to X B` | L1 OOM | Reduce channels or batch size |
| `DRAM Auto slice could not find valid slice configuration` | Large dilated conv can't be sliced | Avoid large dilation or reduce spatial dims |
| `Shardy propagation only supports ranked tensors with a static shape` | Dynamic shape | Fix all input shapes to be fully static |
| PCC < 0.99 | Numerical divergence | Check op replacements that change semantics |

---

## Step 7 — Promote to EXPECTED_PASSING

Once PCC ≥ 0.99 against the CPU bfloat16 ground truth:

```yaml
  <model_name>/pytorch-<VariantName>-single_device-inference:
    status: EXPECTED_PASSING
```

Remove `bringup_status` and `reason` fields.

---

## Known tt-mlir limitations (as of 2026-03)

| Op | Issue | Workaround |
|---|---|---|
| `MaxPool3d` | No bfloat16 support | Replace with stride-2 `Conv3d` |
| `ConvTranspose3d` (lhs_dilation) | Incorrect result in tt-mlir | `F.interpolate` + `Conv3d(1×1×1)` |
| `Conv2d` dilation=18, 512→512ch, width=44 | DRAM slice fatal | Needs compiler fix |
| `Conv3d` out_channels=1280 | L1 static CB overflow | Needs compiler fix |
| Dynamic token shapes | Shardy requires static shapes | Not supported for LLMs |
| Models > single chip DRAM | Out of memory | Multi-chip or not supported |

---

## Quick reference

| What | Where |
|---|---|
| Model source | `third_party/tt_forge_models/<model>/pytorch/src/model.py` |
| Model loader | `third_party/tt_forge_models/<model>/pytorch/loader.py` |
| Test config | `tests/runner/test_config/torch/test_config_inference_single_device.yaml` |
| CPU vs TT verify script | `scripts/verify_model_cpu_vs_tt.py` |
| Saved ground truth tensors | `tests/torch/graphs/<model_name>_*.pt` |
| Run model test | `pytest tests/runner/test_models.py -k <model_name>` |
