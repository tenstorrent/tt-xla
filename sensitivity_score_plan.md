

# Sensitivity Score Calculator — Implementation Plan

## Goal

Script that computes per-tensor sensitivity scores `S(T)` to measure how much quantizing each
weight tensor to BFP4 degrades the model. Model-agnostic, configurable via CLI.

## Formula

```
S(T) = SUM_i [ Fii * (wi - Q(wi))^2 ]

Fii = (1/D) * SUM_d [ g[d,i]^2 ]
```

- `wi` — original weight value at index i
- `Q(wi)` — BFP4-quantized weight value (via `quantize_via_ttnn`)
- `g[d,i]` — gradient of cross-entropy loss w.r.t. `wi` on sample `d`
- `D = 100` samples from C4

## Design Decisions


| Decision           | Value                                                                                              |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| Model              | CLI arg (`--model`, e.g. `meta-llama/Llama-3.2-3B`)                                                |
| Quantization dtype | `ttnn.bfloat4_b`                                                                                   |
| Dataset            | C4, `en` split, 100 samples                                                                        |
| Sequence length    | 128 (fixed)                                                                                        |
| Target tensors     | All `nn.Linear` `.weight` params (2D, safe for `quantize_via_ttnn`)                                |
| Device             | `ttnn` device opened via `ttnn.open_device(device_id=0)`, CLI arg `--device-id` (int, default `0`) |
| Compute            | Forward/backward on CPU; TT device used only for quantization                                      |
| Output             | Sensitivity scores sorted descending, saved to JSON                                                |


## To-Do List

1. **Setup: load model and tokenizer**
  - Accept `--model` as CLI arg (HuggingFace model name or local path)
  - Load model in bf16/fp32, keep on CPU for gradient accumulation
  - Load corresponding tokenizer
  - Open TT device: `device = ttnn.open_device(device_id=args.device_id)`, close in `finally`
2. **Setup: load C4 dataset**
  - Stream 100 samples from `allenai/c4`, `en` split
  - Tokenize each sample, truncate/pad to 128 tokens
3. **Collect target weight tensors**
  - Walk all `nn.Linear` modules in the model (matmul weights)
  - Collect `(name, param)` pairs for `.weight` parameters only
4. **Forward/backward pass loop — compute Fii per tensor**
  - Initialize a gradient accumulator (zero tensor, same shape) for each target weight
  - For each of D=100 samples:
    - Run forward pass on device
    - Compute cross-entropy loss: `loss = CE(logits.view(-1, vocab_size), labels.view(-1))`
    - Run `loss.backward()`
    - For each target weight: accumulate `param.grad ** 2` into its buffer
    - Zero gradients before next sample
  - After loop: divide each buffer by D → `Fii` per element per tensor
5. **Compute quantization error — `(wi - Q(wi))^2`**
  - For each target weight tensor `T`:
    - Call `quantize_via_ttnn(T.data, dtype=ttnn.bfloat4_b, device=device)` to get `Q(T)`
    - Compute elementwise squared error: `(T.data - Q(T)) ** 2`
6. **Compute sensitivity score S(T)**
  - `S(T) = (Fii * quant_error).sum()` — elementwise product, then reduce
  - Store results as `{tensor_name: float(S(T))}` dict
7. **Output results**
  - Sort by sensitivity score descending
  - Save full results to `sensitivity_scores.json` (or path via `--output` arg)

---

  **- How does SEQ_LEN affect final results?**

Cross-entropy averages over all token positions. With longer sequences, early tokens (which have less context) are diluted by   later tokens (which have full context), so gradients skew toward weights that matter for well-contextualized predictions.

Practical impact for sensitivity scoring:

- Short SEQ_LEN (128) — gradients are noisier, early-token behavior dominates, may underestimate sensitivity of attention layers
- Longer SEQ_LEN (512-1024) — more stable Fii estimates, better reflects real inference conditions

---

  **- How to make backward pass faster?**

**CPU optimizations:**
- Freeze all non-linear-weight params (`requires_grad_(False)`) — reduces gradient allocation and some backward compute ✅
- Reduce `SEQ_LEN` (e.g. 64) — smaller activations, faster backward
- Reduce `NUM_SAMPLES` — fewer backward passes; Fisher approximation degrades gracefully
- `zero_grad(set_to_none=True)` — avoids zeroing tensors, slightly faster per iteration
- `torch.compile(model)` — 20–40% speedup per sample after one-time compile cost; worth it at ≥100 samples

**GPU optimizations:**
- Move model and inputs to CUDA (`model.to("cuda")`) — single biggest lever, order-of-magnitude speedup
- `torch.compile` — more impactful on GPU than CPU due to kernel fusion
- Flash attention — enabled by default in recent PyTorch; speeds up attention backward significantly
