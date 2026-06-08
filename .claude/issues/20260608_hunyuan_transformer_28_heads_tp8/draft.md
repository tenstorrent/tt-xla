> **Filing note:** the `(1,8)` mesh in this repro is a **fundamentally invalid** tensor-parallel configuration for this model — `num_attention_heads = 28` is not divisible by 8. The underlying compile failure is therefore *expected*. The actionable bug here is the **diagnostic quality**: an invalid TP degree should fail with a clear "TP degree must divide num_attention_heads" message, not a cryptic downstream `reshape` element-count mismatch (`14680064` vs `1835008`, exactly 8×). Consider whether to file as a usability/validation `enhancement` rather than a `bug`. See § Notes.

### Describe the bug

- Model: **HunyuanImage 2.1 transformer** (`HunyuanImageTransformer2DModel`, MM-DiT, 17.45B), test node `tests/torch/models/HunyuanImage_2_1/test_transformer.py::test_transformer_sharded`.
- Arch: `n300-llmbox` (Wormhole), `TT_XLA_SPMD=1`, `CONVERT_SHLO_TO_SHARDY=1`, mesh **`(1, 8)`** over 8 devices, dtype `fp32`.
- Surface error (PJRT): `ValueError: Error code: 13` → `module_builder.cc:884 ERR| Failed to run stablehlo pipeline`.
- Underlying compiler error: `loc("reshape.568"): error: number of output elements (14680064) doesn't match expected number of elements (1835008)` — a **ratio of exactly 8×**.
- Root cause: an 8-way **model-axis** split tries to shard `num_attention_heads = 28` across 8 devices. `28 % 8 != 0`, so the head-reshape after the partitioner is dimensionally inconsistent. The failure is a *symptom* of an invalid sharding degree, surfaced as an opaque reshape mismatch with no mention of heads or TP degree.

### Call chain

```
test_transformer_sharded                              # mesh (1,8) via loader.get_mesh_config(8)
  → ModelLoader(TRANSFORMER) + load_shard_spec         # shards num_attention_heads (28) across model axis = 8
      → HunyuanImageTransformer2DModel.forward
          → attention head reshape  [..., 28*head_dim] → [..., 8, ...]   # 28 not divisible by 8
              → torch_xla SPMD partitioner → Shardy (CONVERT_SHLO_TO_SHARDY=1)
                  → reshape.568                          # output elems 14680064 != expected 1835008 (8x)
                      → "Failed to run stablehlo pipeline" → ValueError: Error code: 13
```

### Key observations

- `14680064 / 1835008 = 8.0` exactly — the extra factor is the invalid 8-way head split; the partitioner replicates/mis-tiles where it cannot evenly divide 28 heads.
- The error names only `reshape.568` — **nothing** in the message points at the attention-head dimension or the chosen TP degree, making the root cause hard to diagnose without prior knowledge of the model's head count.
- This config is **not viable by construction**: the only pure model-axis TP degrees that divide `28` are 1, 2, 4, 7, 14, 28; of those, only ≤8-chip degrees 2/4/7 are usable on this box. `(1,8)` was an exhaustive-search experiment, not a recommended path.
- Companion experiments (same model, same branch): `(1,4)` bf16 **compiles and runs** (then OOMs on weight pressure); `(2,4)` 2D FSDP fails earlier on `sdy.collective_permute` lowering (tt-mlir#3370, drafted separately).

### Experiments / sanities

| Mesh | dtype | Stage reached | Result |
|------|-------|---------------|--------|
| `(1, 8)` 1D model | fp32 | compile (StableHLO pipeline) | **FAIL** — `reshape.568` 8× element mismatch (this issue; 28 ∤ 8) |
| `(1, 4)` 1D model | bf16 | runtime execution | compiles + runs, then OOM (weight pressure) |
| `(2, 4)` 2D FSDP | fp32 | compile | FAIL — `sdy.collective_permute` not lowered (tt-mlir#3370) |

### Steps to reproduce

```bash
# tt-xla, branch akannan/hunyuan_image_e2e_pipeline
export TT_XLA_ARCH=n300-llmbox
export TT_VISIBLE_DEVICES=0,1,2,3      # SPMD runtime reports 8 devices on the llmbox
export TT_XLA_SPMD=1
export CONVERT_SHLO_TO_SHARDY=1

# Reproduces only when get_mesh_config / load_shard_spec yields a (1,8) model-axis split.
pytest -svv "tests/torch/models/HunyuanImage_2_1/test_transformer.py::test_transformer_sharded"
```

### Logs

- Primary log: `/proj_sw/user_dev/ctr-akannan/3_jun_yyz/tt-xla/.claude/bringup/hunyuan_image_2_1/logs/iter_2_verify_tp.log`
- Diagnosis: `/proj_sw/user_dev/ctr-akannan/3_jun_yyz/tt-xla/.claude/bringup/hunyuan_image_2_1/diagnosis_transformer.json` (iter 2)

Decisive excerpt:

```
loc("reshape.568"): error: number of output elements (14680064) doesn't match expected number of elements (1835008)
2026-06-04 20:32:05.416 ( 179.009s) ... module_builder.cc:884    ERR| Failed to run stablehlo pipeline
Created device mesh: (1, 8) with 8 devices.
FAILED
...
E   ValueError: Error code: 13
```

### Expected behavior

When a tensor-parallel model-axis degree does not evenly divide a sharded dimension (here `num_attention_heads = 28` on an 8-way axis), the framework should fail fast with an explicit, actionable diagnostic — e.g. *"tensor-parallel degree 8 does not divide num_attention_heads=28; choose a degree in {1,2,4,7,14,28}"* — rather than an opaque downstream `reshape` element-count mismatch. Ideally `get_mesh_config` / shard-spec validation rejects the invalid degree before compilation.

### Suggested next steps

1. **Validate TP degree against shardable dims** in `get_mesh_config` / `load_shard_spec` (or in the SPMD partitioner front-end) so a degree that does not divide `num_attention_heads` is rejected up front with a clear message.
2. **Improve the compiler diagnostic**: when a `reshape` element-count mismatch originates from an indivisible sharded dim, annotate the error with the offending dimension and TP degree instead of bare element counts.
3. For this model specifically, **constrain the mesh** to head-divisible degrees (2/4/7); `(1,8)` should never be emitted by the loader.

### Related issues

- **tenstorrent/tt-mlir#3370** — `sdy.collective_permute` lowering gap on the companion `(2,4)` 2D FSDP mesh for the same transformer. Distinct root cause from this `(1,8)` head-divisibility failure (compile-time reshape mismatch, not collective_permute).
- No similar issues found in `tenstorrent/tt-xla` at time of investigation.

### Notes

- Arch: `n300-llmbox` (Wormhole, 12 GiB DRAM/chip). Classification: **compile-time**, but the trigger is an **invalid sharding configuration**, not a lowering gap — severity is low; primary value is diagnostic quality + front-end validation.
- This may fit better as an `enhancement` (input validation / error-message quality) than a `bug`. Adjust the label before filing.
- The HunyuanImage 2.1 transformer bringup is ESCALATED in tt-xla; this `(1,8)` experiment is one of three exhaustive-search iterations (`diagnosis_transformer.json`).
- tt-xla issues need **Type: Bug** set in the GitHub UI (gh CLI cannot set it).
