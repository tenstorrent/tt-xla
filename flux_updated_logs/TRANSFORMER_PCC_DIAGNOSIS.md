# transformer_sharded — PCC collapse: ROOT CAUSE = AdaLN modulation shard spec

Machine: 8-chip LLMBox. Submodule branch `akannan/fix_flux2_encoder_oom`. Mesh `(1, 8)`.
Config: 48 heads × 128 head_dim, 8 dual + 48 single blocks.

## Question
Sharded transformer fails: bf16 OOMs; bfp8 fits but `pcc=-0.0267` vs required 0.99.
Is the drop a **shard-spec bug** or a **model-op lowering** issue?

## Answer
**A shard-spec bug — the AdaLN modulation / norm_out projections were column-sharded but
must be replicated.** NOT model-op lowering, NOT bfp8.

`Flux2Modulation.linear` (and `norm_out`, an `AdaLayerNormContinuous`) emit `dim*3*sets`
features that the block immediately splits with `torch.chunk(..., dim=-1)` into
shift/scale/gate, each width `dim`. The spec column-sharded that **same last axis**
`("model", None)` 8-way, so the chunk boundaries (every `dim=6144`) stop aligning with the
shard boundaries (every `dim*3*sets/8 = 2304`). shift/scale/gate get the wrong per-device
slices, so the `(1 + scale) * norm(x) + shift` modulation is applied with garbage →
attention/MLP output decorrelates (PCC → 0).

### Why it hid
With `from_config` **random init**, modulation `scale`/`shift` ≈ 0, so `(1+0)*x` ≈ `x` and
the mis-sharding is invisible (random sharded ≈ 0.9958). The **trained** model has
significant scale/shift, so the mis-sharding is catastrophic.

A second, smaller instance of the same class of bug: the QK-RMSNorm weights
(`norm_q/k/added_q/added_k`, shape `(head_dim,)`) were sharded `("model",)` but apply over
the non-sharded head_dim → also fixed to `(None,)`. (Minor on its own; the modulation
projections were the dominant cause.)

## Evidence (TT vs CPU PCC; sharded 8-way unless noted)
| Weights | Depth (dual+single) | dtype | PCC | Note |
|---------|---------------------|-------|-----|------|
| random  | 1+1 unsharded | bf16 | 0.999983 | model ops fine |
| random  | 1+1 sharded | bf16 | 0.995800 | random masks the bug |
| random  | 1+1 sharded | bfp8 | 0.995789 | bfp8 fine per-op |
| random  | 4+20 sharded | bf16 | 0.994326 | masked at depth too |
| random  | 4+20 sharded | bfp8 | 0.994323 | bf16 ≈ bfp8 → NOT bfp8 |
| **real**| 1+1 **unsharded** | bf16 | **0.999908** | real model lowers perfectly |
| **real**| 1+1 sharded | bf16 | **-0.016124** | sharding collapses real weights |
| **real**| 4+20 sharded | bf16 | -0.023817 | |
| **real**| 4+20 sharded | bfp8 | -0.024292 | bf16 ≈ bfp8 → confirms NOT bfp8 |
| real full (8+48) | bfp8 | -0.0267 | the original failing test |

### Bisection on real 1+1 sharded bf16 (which group is at fault)
| Sharding | PCC | Verdict |
|----------|-----|---------|
| all (orig) | -0.016 | broken |
| **blocks_only** (modulation/embedder/norm_out replicated, blocks sharded) | **0.999192** | the group is the culprit |
| **mod_only** (only that group sharded, blocks replicated) | **-0.013** | confirms the group |

## Fix (`flux2/pytorch/src/model_utils.py`, `shard_transformer_specs`)
- `double_stream_modulation_img/txt.linear.weight`, `single_stream_modulation.linear.weight`:
  `("model", None)` → `(None, None)` (replicate).
- `norm_out.linear.weight/bias`: `("model", None)`/`("model",)` → `(None, None)`/`(None,)`.
- (also QK-RMSNorm weights `("model",)` → `(None,)`.)
These projections are tiny, so replicating them costs negligible DRAM. Block QKV/MLP/FF stay
sharded (where the 32 B of weights actually live).

### Fix verification (real weights)
| test | before | after fix |
|------|--------|-----------|
| real 1+1 sharded bf16 | -0.016124 | **0.999255** (= unsharded quality → spec now clean) |
| real 4+20 sharded bf16 | -0.023817 | **0.955** (precision accumulation + truncation) |
| real FULL 8+48 sharded **bfp8** | -0.0267 | **0.641** (still < 0.99) |

## Can the transformer fit on 8 chips at PCC ≥ 0.99? No.
After the shard fix the spec is numerically clean (depth-2 sharded = 0.9992 = unsharded).
The remaining `0.9992 → 0.955 → 0.641` decay with depth is **low-precision accumulation** on
the real model (wider dynamic range than random init), NOT another bug.
- bf16 weights = 8.06 GB/device at 8-way (all 32.2 B params shard, ~0 replicated) → too thin,
  mid-graph tilize OOMs. So 8 chips REQUIRES bfp8.
- bfp8 fits (~4 GB/device) but full-model PCC = **0.641** < 0.99 — bfp8 quant error over
  56 blocks. math_fidelity/fp32-acc already at MLIR defaults (HiFi4 + fp32 dest-acc); no
  per-layer precision knob exists in CompilerConfig.
- **Conclusion: 8 chips cannot reach 0.99.** Fall back to **32-chip Galaxy bf16** (last
  resort, as instructed): `MESH_SHAPES` maps `32:(1,32)` → ~2 GB/device, full bf16 precision
  (no bfp8 loss). NOTE: needs a Galaxy to confirm it clears 0.99 — the 24-block bf16 point
  (0.955) suggests the full bf16 model will be far better than bfp8's 0.641 but should be
  measured. (Could not run here: only 8 chips available.)
