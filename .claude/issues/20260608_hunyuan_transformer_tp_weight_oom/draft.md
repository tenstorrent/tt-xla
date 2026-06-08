> **⚠️ DUPLICATE — do not file as a new issue.** This failure is already tracked by
> **#4780 — [HunyuanImage-2.1-Distilled-Diffusers] OOM in transformer**. The body below is
> drafted as a **comment to post on #4780**, contributing (a) the structured TP-mesh iteration
> matrix from single-chip→multichip bringup and (b) the upstream tt-mlir#3370 linkage.
>
> It also **corrects our internal bringup diagnosis** (`diagnosis_transformer.json` called this
> *weight-bound*; #4780's analysis shows the 771 MB buffer is the unfused-SDPA score matrix — a
> *reducible activation*). See "Diagnosis reconciliation" below.

---

### Supplemental bringup data for #4780 (HunyuanImage-2.1 transformer TP OOM)

Adding the tensor-parallel iteration matrix from the `hunyuan_image_2_1` transformer bringup on `n300-llmbox` (Wormhole, 12 GiB DRAM/chip). The 4-chip OOM here is the **same `771162112 B` buffer** already reported in this issue.

#### TP-mesh iteration matrix

| Iter | Mesh | dtype | Stage | Outcome |
|------|------|-------|-------|---------|
| 1 | `(2,4)` | fp32 | compile | `sdy.collective_permute` not lowerable + "op requires same type for all operands and results" → **tt-mlir#3370** |
| 2 | `(1,8)` | fp32 | compile | `reshape.568`: output elements `14680064` vs expected `1835008` (8×) — **8-way model-axis invalid: `num_attention_heads=28` not divisible by 8** |
| 3 | `(1,4)` | bf16 | runtime exec | compiled + executed 45 graphs, then **`TT_FATAL` OOM**: `771162112 B` DRAM buffer, `free: 72122304 B` (chip ~full), on `ttnn::add` (`BinaryNgDeviceOperation`) |

#### Diagnosis reconciliation (correction to our bringup notes)

Our bringup `diagnosis_transformer.json` classified iter-3 as **weight-bound** — "8.73 GiB/chip weights leave ~2–3 GiB headroom; a 771 MB `ttnn.add` tipped over; OOM is weight-pressure, NOT reducible activation."

**This issue's analysis supersedes that**: the 771 MB buffer is the **native-SDPA `attn_weight + mask` add** materializing a `[1, 7, 5224, 5224]` f32 joint-attention score matrix (`7 × 5224² × 4 B ≈ 764 MB ≈ 771162112 B`). The SDPA composite is skipped because the `attn_mask` is bool/`i1`. Our own iter-3 backtrace is consistent with this — the aborting op is `BinaryNgDeviceOperation` → `ttnn::add` (the scores+mask add), **not** a weight load:

```
 --- ttnn::operations::binary_ng::BinaryNgDeviceOperation::create_output_tensors(...)
 --- ttnn::prim::binary_ng(...)
 --- ttnn::add(...)
 --- tt::runtime::ttnn::operations::eltwise::binary::run(EltwiseBinaryOp ...)
```

**Implication:** the 771 MB is a *reducible activation*, not irreducible weight pressure. The fix path in this issue (engage the fused TTNN SDPA so the score matrix is never materialized) is the right lever; reducing weight footprint alone (e.g. a hypothetical 8-way) would not remove this buffer.

#### Upstream blocker for the 2D weight-sharding path

The `(2,4)` 2D FSDP-style reshard (which would halve per-chip weights to ~4.4 GiB) is blocked on **tt-mlir#3370** — `shardy collective permute to stablehlo collective permute` (`sdy.collective_permute` not lowered). Same lowering gap also seen in #5040 (FLUX.2-dev).

#### Why no pure-model-TP degree fits

- `8-way` (would fit at 4.36 GiB/chip) is **invalid**: `28 heads % 8 ≠ 0`.
- `4-way` is head-valid and compiles/runs but OOMs (above).
- `7-way (1,7)` is head-valid (`28/7=4`) and weight-fits (`34.9/7 = 4.99 GiB/chip`) but is **untested** — non-power-of-2 mesh, 1 idle chip on n300-llmbox, collective topology mapping unverified.

### Steps to reproduce (iter-3, the 4-chip OOM)

```bash
git checkout akannan/hunyuan_image_e2e_pipeline
TT_XLA_ARCH=n300-llmbox TT_VISIBLE_DEVICES=0,1 TT_XLA_SPMD=1 CONVERT_SHLO_TO_SHARDY=1 \
  pytest -svv "tests/torch/models/HunyuanImage_2_1/test_transformer.py::test_transformer_sharded"
```

Decisive line:

```
2026-06-04 20:51:56.334 | critical | Always | TT_FATAL: Out of Memory: Not enough space to allocate
  771162112 B DRAM buffer across 12 banks, where each bank needs to store 64266240 B,
  but bank size is 1071821792 B (allocated: 999699488 B, free: 72122304 B,
  largest free block: 49191520 B) (assert.hpp:104)
```

### Logs

- 4-chip `(1,4)` bf16 runtime OOM: `.claude/bringup/hunyuan_image_2_1/logs/iter_3_verify_tp.log` (decisive line **25 / 29**; `binary_ng`/`ttnn::add` backtrace lines **40–44**)
- Bringup diagnosis JSON (the one this comment corrects): `.claude/bringup/hunyuan_image_2_1/diagnosis_transformer.json`

### Related issues

- **#4780** — primary issue this duplicates / supplements (HunyuanImage transformer OOM).
- **tenstorrent/tt-mlir#3370** — `sdy.collective_permute` lowering; blocks the `(2,4)` 2D weight-sharding path.
- **#5040** — FLUX.2-dev transformer hits the same Shardy `collective_permute` lowering gap.
- **#4761** — [HiDream-I1-Fast] OOM in transformer (related transformer-OOM class).

### Notes

- Arch: `n300-llmbox` (Wormhole, 12 GiB/chip). Branch: `akannan/hunyuan_image_e2e_pipeline`.
- This is a **comment supplement**, not a new issue — the `gh-command.sh` in this folder runs `gh issue comment 4780`, not `gh issue create`.
