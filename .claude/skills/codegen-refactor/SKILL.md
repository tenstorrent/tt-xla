---
name: codegen-refactor
description: "Use when refactoring a codegenerated TTNN model (from tt-xla codegen_py pipeline) into clean, modular, human-readable code with LightweightModule classes. Triggers on: 'codegen', 'codegenerated model', 'refactor generated code', or directories with large main.py files containing patterns like main_const_eval_N and utils.load_tensor."
---

# Codegen-to-Clean TTNN Model Refactoring

Use when you have a codegenerated TTNN model (from tt-xla's `codegen_py` pipeline) and need to refactor it into clean, maintainable, human-readable code with LightweightModule classes.

## When to trigger

- User mentions "codegen", "codegenerated model", "refactor generated code"
- A directory contains a large `main.py` (10K+ lines) with patterns like `main_const_eval_N`, `utils_constEvalFuncWrapper`, `ttnn_matmul_N`, `utils.load_tensor("./tensors/argN.tensorbin")`
- User asks to clean up or make a model readable/maintainable

## Overview

The codegen pipeline (`tt_torch/codegen.py`) produces a single monolithic Python file with:
- All operations inlined (no loops, no modules)
- Anonymous weight files (`arg0.tensorbin` through `argN.tensorbin`)
- Duplicated const-eval functions (often 85+ with only ~10 unique patterns)
- No model structure — just a flat sequence of TTNN ops

The goal is to produce clean code matching the PyTorch model structure:
- `LightweightModule` classes (Attention, FeedForward, TransformerBlock, etc.)
- Weights loaded from HuggingFace at runtime (no `.tensorbin` files)
- Repeated blocks use loops
- Deduplicated const-eval functions
- PCC > 0.99 against PyTorch reference

## Reference example

The Z-Image transformer refactoring in `z-transformer-only/` is the golden reference:
- `model_ttnn.py` (~1.1K lines) — from 35K-line codegen
- `model_pt.py` — standalone PyTorch reference
- `consteval.py` — deduplicated const-eval patterns
- `tensor_load_config.json` — per-weight layout/dtype/device config
- `main.py` — entry point with PCC verification
- `parse_mlir_args.py` — MLIR arg-to-weight-name parser

Also see the CLIP resampler on branch `svuckovic/sdxl-clip-models` in `clip_resampler_codegen_1/`.

## Process (12 steps)

### Step 1: Parse MLIR for arg-to-weight mapping

The MLIR file (`ttnn.mlir`) contains `ttir.name` annotations mapping each `%argN` to its PyTorch weight name. Copy `parse_mlir_args.py` from this skill's directory into the model directory and run it:

```bash
cp .claude/skills/codegen-refactor/parse_mlir_args.py <model-dir>/
cd <model-dir>
python3 parse_mlir_args.py --mlir ttnn.mlir --output arg_mapping.json --verbose
```

This produces `arg_mapping.json`: `{"0": {"name": "final_layer.linear.bias", "shape": [64], "dtype": "bf16"}, ...}`

### Step 2: Establish baseline PCC

Run the existing codegen model and record PCC against a PyTorch reference output. This is the target to maintain throughout refactoring. Modify the codegen's `main()` to compute PCC:

```python
ref = torch.load("path/to/reference_output.pt", weights_only=True)
out_tt = ttnn.to_torch(ttnn.from_device(output))
pcc = calculate_pcc(ref, out_tt)  # Must be > 0.99
```

### Step 3: Create `model_pt.py`

Standalone PyTorch reference model. Copy all building blocks from the original PyTorch model file — do NOT import from other directories. Include:
- All nn.Module classes (copy verbatim)
- `ModelNamePT(nn.Module)` wrapper that loads from HuggingFace with local caching
- `get_input()` function returning sample inputs with caching
- `state_dict_for_ttnn()` method for weight extraction

### Step 4: Create `utils.py`

Minimal — just `calculate_pcc()`:

```python
def calculate_pcc(x, y):
    x_flat, y_flat = x.flatten().float(), y.flatten().float()
    vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
    denom = vx.norm() * vy.norm()
    return torch.tensor(float("nan")) if denom == 0 else (vx @ vy) / denom
```

### Step 5: Analyze const-eval patterns

Read all `main_const_eval_N` functions (typically lines 1 to `_main`). Categorize into unique patterns. Common patterns:

| Pattern | Operations | Typical use |
|---------|-----------|-------------|
| Transpose+cast | `to_device → TILE → permute([1,0]) → FLOAT32` | 2D weight matrices for `ttnn.linear` |
| Cast only | `to_device → TILE → FLOAT32` | 1D biases |
| Attn mask | `full(-inf) → full(0) → where → repeat` | Attention masks |
| Scalar ones | `full([1,1], 1.0)` | adaLN `1 + scale` |
| Reshape buffer | `to_device → TILE → reshape` | Special buffers |
| RoPE precompute | Complex 100+ op sequence | Position embeddings |

### Step 6: Generate `tensor_load_config.json`

Using the arg mapping (Step 1) and const-eval analysis (Step 5), generate a `tensor_load_config.json` that specifies `{layout, dtype, on_device}` for every weight. This drives `load_weights_from_pytorch` — no hardcoded name patterns.

```json
{
  "layers.0.attention.to_q.weight": {"layout": "TILE", "dtype": "BFLOAT16", "on_device": true},
  "layers.0.adaLN_modulation.0.bias": {"layout": "ROW_MAJOR", "dtype": "BFLOAT16", "on_device": false},
  "cap_pos_ids": {"layout": "ROW_MAJOR", "dtype": "INT32", "on_device": false}
}
```

Rules for classification:
- **2D weight matrices** used in matmul/linear: `TILE, BFLOAT16, on_device=true`
- **RMSNorm weights**: `TILE, BFLOAT16, on_device=true`
- **Pad tokens**: `TILE, BFLOAT16, on_device=true`
- **Position IDs**: `ROW_MAJOR, INT32, on_device=false`
- **Bool masks**: `ROW_MAJOR, BFLOAT16, on_device=false` (processed by consteval)
- **All other weights** (biases, RoPE tables, embedder params): `ROW_MAJOR, BFLOAT16, on_device=false` (processed by consteval)

Also extract const-eval wrapper mappings from `_main()` to understand which const-eval pattern is applied to each weight — this informs `consteval.py`'s `run_const_evals()`.

### Step 7: Extract op sequences from codegen

For each module type, find a representative instance in `_main()` and extract the exact TTNN op sequence. Key modules:
- **Attention**: Q/K/V projections, QK-norm, RoPE application, SDPA, output projection
- **FeedForward**: w1+silu, w3, element-wise multiply, w2
- **TransformerBlock**: adaLN modulation + attention path + FFN path + residuals
- **Embeddings**: Patch embed, timestep embed, caption embed
- **Final layer**: adaLN + norm + linear + un-patchify

Pay attention to:
- `memory_config` parameters (typically all DRAM interleaved)
- `dtype` in each op (FLOAT32 for compute, BFLOAT16 for storage)
- `compute_kernel_config` (WormholeComputeKernelConfig with HiFi4)
- `transpose_a`/`transpose_b` in matmul calls
- `activation` parameter (e.g., "silu" fused into matmul)
- `ttnn.deallocate` calls

### Step 8: Build `consteval.py`

One function per unique pattern, plus:
- `prepare_rope_embeddings()` for the complex RoPE precomputation
- `run_const_evals(weights, device)` master function that applies the right transformation per weight name

### Step 9: Build `model_ttnn.py`

LightweightModule classes with exact ops from Step 7. Key constants at top:
```python
DRAM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
COMPUTE_CONFIG_NORM = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True)
```

Include `load_weights_from_pytorch(state_dict, device)` that reads `tensor_load_config.json` and applies the correct dtype, layout, and device placement per weight — no hardcoded name patterns.

```python
def load_weights_from_pytorch(state_dict, device):
    config = json.load(open(Path(__file__).parent / "tensor_load_config.json"))
    weights = {}
    for name, tensor in state_dict.items():
        t = ttnn.from_torch(tensor)
        cfg = config.get(name, {"layout": "ROW_MAJOR", "dtype": "BFLOAT16", "on_device": False})
        t = ttnn.to_dtype(t, DTYPE_MAP[cfg["dtype"]])
        if cfg["layout"] == "TILE":
            t = ttnn.to_layout(t, ttnn.Layout.TILE)
        if cfg["on_device"]:
            t = ttnn.to_device(t, device=device, memory_config=DRAM)
        weights[name] = t
    return weights
```

### Step 10: Build `main.py`

Entry point: load PyTorch model, get reference output, load TTNN model, run both, compare PCC.

### Step 11: Debug iteratively

Run the model. Fix errors one by one:
1. Runtime errors (host vs device, dtype, shape mismatches)
2. PCC mismatches (compare weights, then intermediate tensors)

### Step 12: Cleanup

Remove codegen artifacts (`main_codegen.py`, `tensors/`, etc.). Keep MLIR files. Commit.

## Lessons learned (gotchas)

| Issue | Detail |
|-------|--------|
| **Weight transpose** | Codegen const-eval transposes 2D weights with `permute([1,0])`, then matmul uses `transpose_b=True`. Net effect = PyTorch layout. Keep both or change both. |
| **RoPE cos/sin ordering** | Concatenation order in RoPE tables may not match variable names. Always verify slicing against codegen. |
| **`LightweightModule` not callable** | Use `.forward()` explicitly, not `model(inputs)`. |
| **Bool tensors** | Can't be tilized. Cast to BFLOAT16 before `to_layout(TILE)`. |
| **Int32 tensors** | Position IDs are INT32. Keep on host for embedding lookups. |
| **Inputs must be on device** | Move all inputs to device at start of `forward()`. |
| **`ttnn.linear` vs `ttnn.matmul`** | `linear` expects pre-transposed weight + optional bias. `matmul` with `transpose_b=True` transposes at runtime. |
| **Manual LayerNorm** | Some models use manual mean/rsqrt instead of `ttnn.layer_norm`. Check codegen. |
| **SDPA calls per block** | Don't assume — check codegen. Could be 1 call (all heads) or N calls (head groups). |
| **Shared tensors across layers** | Some tensors (RoPE embeddings, adaln_input) are computed once and reused across all layers. Identify these. |
| **t_embedder intermediate** | The MLP intermediate (before final projection) may be kept alive for the final layer's adaLN. |

## Quality criteria

- [ ] PCC > 0.99 against PyTorch reference
- [ ] All weights loaded from HuggingFace (zero .tensorbin files)
- [ ] Repeated blocks use loops, not inline copies
- [ ] Clean LightweightModule class hierarchy mirroring PyTorch
- [ ] Const-eval functions deduplicated
- [ ] Directory is self-contained (no cross-directory imports)
- [ ] Model runs with `./run` script
