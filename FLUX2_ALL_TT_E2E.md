# FLUX.2-dev on Tenstorrent — full bring-up context (transformer OOM → all-on-TT e2e image)

This documents the path from the first transformer OOM to a working composite
pipeline that runs **all three components on Tenstorrent** and produces a
prompt-faithful 1024×1024 image, plus the spec/branch findings discovered along
the way.

Prompt (fixed, seed 42): *"Realistic macro photograph of a hermit crab using a
soda can as its shell, partially emerging from the can, on a sunlit beach."*

## Timeline / findings

### 1. Transformer (denoiser) OOM at bring-up
The ~32B `Flux2Transformer2DModel` did not fit per-device; the sharded run died
with DRAM OOM during execution (weights nearly fill per-bank DRAM, then a
mid-graph activation tilize/all-gather pushes it over). Pure weight-sharding alone
was not enough.

### 2. bf16 / bfp8 did not fix OOM on the n300 LLMBox (8-chip T3K)
On the 8-chip Wormhole LLMBox, lowering precision did not resolve it:
- full transformer **bf16 PCC ≈ 0.650**, **bfp8 PCC ≈ 0.641** — essentially equal,
  i.e. the accuracy problem was **not** dtype-driven.
- `test_transformer_sharded` still DRAM-OOM-bound even with 8-way `(1, 8)`
  weight-sharding (OOM on a `tilize` / `TilizeWithValPadding` activation, not the
  weights). So more weight-sharding did not buy enough activation headroom there.

### 3. Moved to 4-chip Blackhole — PCC drop was layer-wise, not global
On 4×Blackhole the transformer ran e2e but with a **PCC drop localized to the deep
single-stream blocks** — traced to a **TT device numerical error in those blocks**,
*not* the model, dtype, or the mesh. A depth sweep (`pcc_sweep` / `pcc_zoom`)
isolated the regression to specific single-block depths rather than a uniform loss.

### 4. Root cause = the single-stream shard spec; fixed by the shared-commit spec
The deep single-stream blocks fuse `[Q | K | V | MLP]` into one
`to_qkv_mlp_proj` and the FFN is GEGLU (`linear_in` → chunk/gate). **Column-parallel**
sharding (`("model", None)`) splits those fused outputs *across* the shard
boundary, so the downstream `chunk`/`split`/GEGLU-gate operate on partial tensors
→ garbage (PCC ≈ 0.65) **and** a 2 GB replicated all-gather buffer.

The **shared commit `9177e4ea14`** uses **contraction-parallel** sharding instead
(`(None, "model")` for the fused proj and both FFN linears): the matmul becomes a
partial-sum + all-reduce whose output is fully replicated, so every downstream
slice/chunk/gate sees the complete tensor. Adopting that spec:
- **PCC regained: 0.9884** on the standalone transformer (4×BH, packed seq 4096).
- passes **on latest `main`**; the composite denoiser produces a prompt-faithful image.

> The bad column-parallel spec lives on `akannan/fix_flux2_transformer_oom`
> (and the 2-D (4,8) debug branch); the good contraction spec is on `9177e4ea14`.

### 5. Text-encoder OOM fix (PR #732)
The 24B Mistral3 text encoder OOM'd when its shard spec didn't actually apply —
the descent into `Mistral3ForConditionalGeneration` must reach
`model.language_model` (the module owning `.layers`). PR #732's
`_resolve_text_transformer` (tries `language_model` before `model`, no fixed depth
bound, `visited` guard) makes the spec apply: **362 tensors shard**, encoder runs
on TT, no OOM. Verified here.

> Note: PR #732's branch (`akannan/fix_flux2_encoder_oom`) also carries the *bad*
> column-parallel transformer spec. The working combination is **`9177e4ea14`'s
> good transformer spec + PR #732's encoder descent** — applied together in the
> submodule for this pipeline.

### 6. All-three-on-TT composite → realistic 1024×1024 output ✅
`composite_all_tt.py` runs every compute component on device:
- text encoder (Mistral3 24B) → TT, tensor-parallel sharded
- denoiser (32B) → TT, tensor-parallel sharded (the good contraction spec)
- VAE decoder (~84M) → TT, replicated

Memory-smart sequential design so peak DRAM ≠ sum: **Stage 1** places the encoder
on device, encodes the prompt, then frees it; **Stage 2** runs denoiser + VAE
(`pipe(prompt_embeds=...)` skips re-encoding; VAE is placed lazily at decode time
so it doesn't inflate the denoise-loop peak). Result on `main` @ 1024/4-steps:
**`COMPOSITE (ALL-TT) SUCCESS`** — prompt-faithful hermit-crab image.

### 7. Branch-vs-main 1024 denoiser-OOM anomaly (open)
A controlled experiment (same 4×BH device, **no reset between runs**, fresh
kernel cache) showed the 1024 denoiser allocation flips purely on the **tt-xla
branch checkout**:

| State (submodule = `9177e4ea14`) | 1024 denoiser |
|---|---|
| tt-xla `main` | ✅ SUCCESS |
| `akannan/bringup_flux2` (even with **pristine** submodule) | ❌ DRAM OOM |
| tt-xla `main` again | ✅ SUCCESS |

OOM detail: needs a `254 MB`/bank contiguous buffer (2.04 GB across 8 banks) but
the largest free block is only ~`182–187 MB` once the ~4 GB/bank of weights are
placed — a fragmentation cliff right at the edge. **No runtime `.py` file differs
between `main` and the branch** (only `.gitignore` + the submodule *pointer*;
`third_party/tt-mlir` is CMake-fetched, identical), so the mechanism is an
unexplained workspace-state effect on compile-time allocation. The encoder edit is
ruled out (branch + pristine submodule also OOMs). **Use `main` for 1024.**

## How to reproduce (on tt-xla `main`, 4×Blackhole)

```bash
source venv/activate
# submodule must have: 9177e4ea14 good transformer spec + PR #732 encoder descent
export HF_TOKEN=...
export FLUX2_HEIGHT=1024 FLUX2_WIDTH=1024 FLUX2_STEPS=4
python composite_all_tt.py            # -> composite_all_tt.png
```

Standalone reference (denoiser only on TT, encoder+VAE on CPU):
`third_party/tt_forge_models/flux2/pytorch/test_multichip.py`.

## Notes
- 128×128 runs e2e but produces noise — that's FLUX.2's resolution floor (it needs
  native 1024 for a coherent image), independent of spec/device.
- 8 chips (`MESH_SHAPES[8] = (1, 8)`) would give the denoiser far more per-bank
  headroom (8-way weight sharding) and sidestep the 1024 fragmentation cliff.
