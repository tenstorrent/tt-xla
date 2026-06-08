### Describe the bug

- **Model key:** `hunyuan_image_2_1` — `ModelVariant.VAE` (`AutoencoderKLHunyuanImage` decoder), single device, **p150** (Blackhole), fp32. Test: `tests/torch/models/HunyuanImage_2_1/test_vae_decoder.py::test_vae_decoder`.
- **Surface error (pytest):** `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`. **Underlying device abort:** `TT_FATAL: Out of Memory: Not enough space to allocate 34359738368 B DRAM buffer across 7 banks` during execution.
- This is an **activation/allocation** OOM, **not** a weight-size limit. VAE weights are **0.41 B** and fit n150/p150 comfortably; the decode of a `(1, 64, 64, 64)` latent up to a `(1, 3, 2048, 2048)` image produces a **single 32 GiB intermediate** that no single Blackhole device can hold.
- Same failure class as the sibling **Z-Image** VAE decoder OOM (`#4755`); same model family as the HunyuanImage transformer OOM (`#4780`).

### Call chain

```
test_vae_decoder
  → AutoencoderKLHunyuanImage.decode(z=(1,64,64,64) fp32)   # 32x spatial upsample → (1,3,2048,2048)
      → decoder upsampling stack (large spatial feature maps)
          → StableHLO transpose → TTIR → ttnn.permute
              → PermuteDeviceOperation::create_output_tensors
                  → allocate 34359738368 B (32 GiB) DRAM buffer  ← TT_FATAL OOM
```

### Key observations

- **Allocation that fails:** `34359738368 B` (= exactly **32 GiB**) for one DRAM buffer across **7 banks**; each bank needs `4908535808 B` (~4.57 GiB) but **bank size is `4273390016 B`** (~3.98 GiB). At the failure point: `allocated: 1139768832 B`, `free: 3133621184 B`, `largest free block: 2150985792 B`. The request is ~8× the entire per-bank capacity — it could never fit, irrespective of fragmentation.
- **Failing op is `ttnn.permute`** (`PermuteDeviceOperation::create_output_tensors`), i.e. the OOM is on the *output tensor* of an internal layout transpose in the decoder upsampling stack — not the final image (the final `(1,3,2048,2048)` fp32 sample is only ~48 MiB).
- **Activation-bound, not weight-bound:** 0.41 B params; the peak comes from full-resolution `2048×2048` intermediate feature maps, which scale with output area. This mirrors `#4755` (Z-Image VAE OOM, sub-1B weights, allocation-during-execution).
- Runtime to failure: **~677 s** — the OOM occurs late in execution during the high-resolution upsampling, consistent with peak activation at the top of the decoder.

### Steps to reproduce

```bash
git checkout akannan/hunyuan_image_e2e_pipeline
# single-chip Blackhole p150 host, TT_VISIBLE_DEVICES=0
pytest -svv "tests/torch/models/HunyuanImage_2_1/test_vae_decoder.py::test_vae_decoder"
```

Decisive traceback excerpt:

```
2026-06-06 18:21:23.301 | critical | Always | TT_FATAL: Out of Memory: Not enough space to allocate
  34359738368 B DRAM buffer across 7 banks, where each bank needs to store 4908535808 B,
  but bank size is 4273390016 B (allocated: 1139768832 B, free: 3133621184 B,
  largest free block: 2150985792 B) (assert.hpp:104)
...
 --- ttnn::operations::data_movement::PermuteDeviceOperation::create_output_tensors(...)
 --- ttnn::prim::permute(...)
 --- ttnn::permute(...)
...
E   RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13
FAILED tests/torch/models/HunyuanImage_2_1/test_vae_decoder.py::test_vae_decoder
```

### Logs

- Full log: `.claude/bringup/hunyuan_image_2_1/logs/vae_iter_1_p150.log` (on host)
- Decisive line: **line 17 / 21** — `TT_FATAL: Out of Memory: ... 34359738368 B DRAM buffer across 7 banks`
- Backtrace through `ttnn::permute` → `PermuteDeviceOperation::create_output_tensors`: lines 32–37.
- Surface pytest error: line 103 / 139 — `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`.

### Expected behavior

The VAE decoder should decode a `(1, 64, 64, 64)` latent to a `(1, 3, 2048, 2048)` image on a single Blackhole device without allocating a 32 GiB intermediate buffer. Peak activation should be bounded — e.g. by **tiled decode** so full-resolution feature maps are processed in spatial tiles rather than all at once, and/or by an allocator/layout strategy that does not materialize the entire permuted feature map in DRAM.

### Suggested next steps

1. **Tiled decode (applied, unverified):** `load_vae()` now calls `vae.enable_tiling(tile_sample_min_size=512)` (env-gated `VAE_TILE_SAMPLE_MIN_SIZE=512`). A 512 px tile → 16×16 latent caps peak activation to ~1/16 of the full-frame buffer and matches the real `2048²` pipeline path; both CPU golden and TT run the same tiled module so PCC stays apples-to-apples. **Verify:** CPU sanity forward, then TT re-run. If the tiled path won't compile/execute on TT, drop to `tile_sample_min_size=256`.
2. If tiling alone is insufficient or won't lower, investigate the specific `ttnn.permute` whose output is 32 GiB — confirm whether a smaller-footprint layout (avoid full DRAM materialization of the permuted high-resolution feature map) or sharded memory config is viable.
3. If single-device cannot be made to fit even tiled, evaluate tensor-parallel decode across multiple chips (cf. `#4252` Mochi decoder sharding).

### Related issues

- `#4755` — **Z-Image VAE decoder: TT DRAM OOM at 1280×720 pipeline resolution.** Near-identical class: sub-1B VAE, single chip, allocation-during-execution OOM at full pipeline resolution. This issue tracks the **HunyuanImage** VAE surface; the underlying "VAE decode peak activation exceeds single-device DRAM" pattern is shared.
- `#4780` — **[HunyuanImage-2.1-Distilled-Diffusers] OOM in transformer.** Same model family, different component (transformer is weight-bound/TP; this VAE is activation-bound on a single chip).
- `#4252` — **[Mochi] Hitting DRAM OOM when sharding decoder.** Related decoder OOM; relevant if multi-chip decode is pursued (step 3).

### Notes

- **Arch:** p150 (Blackhole, single chip; n150 skipped — needs Wormhole host). Branch: `akannan/hunyuan_image_e2e_pipeline`.
- **Classification:** model-surface OOM filed on tt-xla; root cause is peak activation footprint of the decoder upsampling stack, surfaced via `ttnn.permute` output allocation.
- **tt-xla UI:** set **Type: Bug** after creation (gh CLI cannot set the Type field).
