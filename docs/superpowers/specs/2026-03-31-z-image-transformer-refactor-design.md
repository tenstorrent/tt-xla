# Z-Image Transformer: Codegen-to-Clean Refactoring Design

## Goal

Refactor the 35,252-line codegenerated `z-transformer-only/main.py` into clean, modular, human-readable TTNN code that:
- Mirrors the PyTorch model structure with `LightweightModule` classes
- Loads weights from HuggingFace (no `.tensorbin` files)
- Deduplicates const-eval functions into a separate file
- Loops over repeated transformer blocks instead of inlining them 34 times
- Maintains **PCC = 1.0000042915** against the reference output throughout every change

## Baseline

| Metric | Value |
|--------|-------|
| Reference source | PyTorch model (`model_pt.py`) at runtime |
| Output shape | `[16, 1, 160, 90]` |
| Baseline PCC | 1.0000042915 |
| Baseline duration | ~158s |
| Model | Tongyi-MAI/Z-Image transformer |

## Model Architecture

```
ZImageTransformer
├── t_embedder: TimestepEmbedder (MLP + precomputed freqs)
├── x_embedder: Linear(64, 3840) — patch embedding
├── cap_embedder: RMSNorm(2560) + Linear(2560, 3840) — caption embedding
├── x_pad_token, cap_pad_token: learnable padding tokens
├── rope_embedder: RealRopeEmbedder (precomputed cos/sin tables)
├── noise_refiner: 2x TransformerBlock (with adaLN modulation)
├── context_refiner: 2x TransformerBlock (without modulation)
├── layers: 30x TransformerBlock (with adaLN modulation)
└── final_layer: FinalLayer (LayerNorm + Linear + adaLN)

TransformerBlock (with modulation)
├── adaLN_modulation: Linear(256, 4*3840=15360)
├── attention_norm1: RMSNorm(3840)
├── attention: Attention
│   ├── to_q, to_k, to_v, to_out: Linear(3840, 3840)
│   └── norm_q, norm_k: RMSNorm(128)
├── attention_norm2: RMSNorm(3840)
├── ffn_norm1: RMSNorm(3840)
├── feed_forward: FeedForward
│   ├── w1: Linear(3840, 10240)
│   ├── w2: Linear(10240, 3840)
│   └── w3: Linear(3840, 10240)
└── ffn_norm2: RMSNorm(3840)
```

Config: dim=3840, n_heads=30, head_dim=128, n_refiner_layers=2, n_layers=30, cap_feat_dim=2560, in_channels=16.

## Target File Structure

```
z-transformer-only/
├── main.py           — Entry point: load models, run TTNN, compare PCC vs PyTorch
├── model_ttnn.py     — TTNN LightweightModule classes (ZImageTransformerTTNN)
├── model_pt.py       — PyTorch reference model (standalone, loads from HF)
├── consteval.py      — Deduplicated const-eval functions
├── utils.py          — PCC calculation, shared utilities
├── arg_mapping.json  — Generated arg index -> weight name mapping (build artifact)
└── parse_mlir_args.py — Script to generate arg_mapping.json from MLIR
```

Files to remove after refactoring is complete:
- `main.py` (original 35K-line codegen — replaced)
- `tensors/` directory (538 `.tensorbin` files — replaced by HF loading)
- `ttnn.mlir` (no longer needed at runtime)
- `irs/`, `generated/`, `__pycache__/` directories

## Approach: Bottom-Up Module Extraction

### Phase 1: Foundation

**Step 1.1: Create `model_pt.py`** — Standalone PyTorch reference model.
- Copy `ZImageTransformer` and all building blocks from `z-image/transformer.py`
- Add model loading from HuggingFace (cache to local `.pt` file)
- Add `get_input()` function that returns sample inputs (latent, timestep, caption)
- Verify it produces the same reference output

**Step 1.2: Create `utils.py`** — Shared utilities.
- `calculate_pcc()` function
- Keep it minimal

**Step 1.3: Create `main.py` scaffold** — Entry point.
- Load PyTorch model, get reference output
- Placeholder for TTNN model (initially runs old codegen `_main`)
- Compare PCC, print results
- Model the structure after the CLIP golden standard `main.py`

### Phase 2: Const-Eval Extraction

**Step 2.1: Analyze const-eval patterns** — Categorize the 85 const-eval functions.
From analysis, there are ~5 unique patterns:
1. `full(1,1)` — creates a scalar ones tensor
2. `to_device → to_layout(TILE) → permute([1,0]) → typecast(FLOAT32)` — weight transpose+cast
3. `to_device → to_layout(TILE) → typecast(FLOAT32)` — weight cast only
4. `to_device → to_layout(TILE) → permute → typecast → reshape → repeat` — bias broadcast
5. Zero-arg const evals (scalar constants)

**Step 2.2: Create `consteval.py`** — Deduplicated const-eval functions.
- One function per unique pattern, parameterized by shape/config
- `run_const_evals(weights, device)` entry point that applies all transformations

### Phase 3: TTNN Module Extraction (Bottom-Up)

This is the core of the refactoring. Extract modules from smallest to largest, verifying PCC at each step.

**Step 3.1: `RMSNorm` wrapper** — Wraps `ttnn.rms_norm` with the exact parameters from codegen.
- Identify the exact `ttnn.rms_norm` call pattern used (epsilon, memory config, etc.)
- Weights: single `weight` tensor per norm

**Step 3.2: `Attention` module** — Self-attention with QK-norm and RoPE.
- Extract one attention block's op sequence from the codegen
- Map: Q/K/V projections (matmul), QK norm (rms_norm), RoPE application, SDPA, output projection
- Weights: `to_q.weight`, `to_k.weight`, `to_v.weight`, `to_out.weight`, `norm_q.weight`, `norm_k.weight`
- This is the most complex module — 136 SDPA calls across 34 blocks = 4 SDPA calls per block (likely head-group splitting by the compiler)

**Step 3.3: `FeedForward` module** — SiLU-gated FFN.
- Extract: `w1(x)` → SiLU → multiply by `w3(x)` → `w2(result)`
- Weights: `w1.weight`, `w2.weight`, `w3.weight`

**Step 3.4: `TransformerBlock` module** — Combines attention + FFN with norms and adaLN.
- Two variants: with modulation (noise_refiner, main layers) and without (context_refiner)
- Extract one block's full sequence, parameterize by weights
- Verify one block matches the codegen output

**Step 3.5: `FinalLayer` module** — Output projection with adaLN.
- LayerNorm + scale modulation + linear projection

**Step 3.6: Embedding modules** — TimestepEmbedder, x_embedder, cap_embedder, RoPE.
- Extract the input processing pipeline

**Step 3.7: `ZImageTransformerTTNN` top-level module** — Compose everything.
- Constructor takes PyTorch state_dict, converts to TTNN weights
- Forward pass loops over blocks instead of inlining
- Weight loading via `load_weights_from_pytorch()` function

### Phase 4: Weight Loading

**Step 4.1: `load_weights_from_pytorch()`** — Convert PyTorch state_dict to TTNN.
- Use `arg_mapping.json` to understand required tensor layouts/dtypes per weight
- For each weight: `ttnn.from_torch()` → appropriate layout/dtype → optionally to device
- Build a `tensor_load_config.json` specifying layout, dtype, on_device per weight (like CLIP golden standard)

**Step 4.2: Remove `.tensorbin` dependency** — All weights come from PyTorch model.

### Phase 5: Integration and Cleanup

**Step 5.1: Wire up `main.py`** — Full pipeline.
- Load PyTorch model from HF (with local caching)
- Create TTNN model from PyTorch state_dict
- Run both, compare PCC
- Multiple iterations to measure perf

**Step 5.2: Final PCC verification** — Confirm PCC = 1.0000042915 with the complete refactored model.

**Step 5.3: Cleanup** — Remove old codegen artifacts.
- Delete `tensors/` directory, old `main.py`, `irs/`, `generated/`

## PCC Verification Strategy

PCC must be > 0.99 and remain stable throughout refactoring. Baseline: 1.0000042915.

- **Reference source**: The PyTorch model (`model_pt.py`) generates reference output at runtime. No dependency on external files.
- **Per-module testing**: After extracting each module, run the full model with just that module replaced and verify PCC hasn't changed.
- **Shortcut for iteration speed**: During development, we can cache the PyTorch reference output locally and verify intermediate tensors at block boundaries rather than running the full 158s model every time.
- **Final validation**: Full end-to-end run with all modules composed, verify PCC.

## Key Risks

1. **Op parameter mismatch**: TTNN ops have many config parameters (memory_config, compute_config, conv_config). Must preserve these exactly from the codegen.
2. **Const-eval ordering**: Some const-evals depend on the device being initialized. The refactored code must call them at the right time.
3. **Head-group splitting in attention**: The codegen splits 30 heads into groups for SDPA. Must replicate this exactly.
4. **Weight format differences**: PyTorch weights may need specific transposes/reshapes that the codegen's const-evals handle. Must replicate in `load_weights_from_pytorch()`.

## Out of Scope

- Performance optimization (acceptable perf changes from weight loading strategy)
- Claude skill creation (will be done after refactoring is complete)
- Changes to files outside `z-transformer-only/`
- Changes to the TTNN runtime or compiler
