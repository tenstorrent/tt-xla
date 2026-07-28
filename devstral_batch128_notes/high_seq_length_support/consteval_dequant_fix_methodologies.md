# Full-depth Devstral-2-123B: the ~79-min silent window and DRAM-OOM are the bf16→bfp8 weight conversion, NOT the fp8 dequant

**Log:** `/data/ssalice/temp/tt-xla/devstral_dptp_test_synced_cpu_false_FULL.log`
**Run:** `test_dptp_devstral[mesh_shape0-1024-True-bfp_bf8]`, 88 layers (full depth), 32-chip BH galaxy, mesh `[4,8]` (DP=4, TP=8), `experimental_weight_dtype=bfp_bf8`.
**tt-mlir tree:** `third_party/tt-mlir/src/tt-mlir` @ `183b2b45d8`.

---

## 0. Bottom line

The full-depth run compiles fine (finishes at **337.6 s**), then spends **78.6 min (4718 s) executing 1062 `load_cached` const-eval functions** that materialise the model weights, and **DRAM-OOMs before a single forward op runs** (0 `Executing operation` lines; OOM at 5056.2 s).

- The **fp8→bf16 dequant is genuinely done host-upfront** by `fp8_dequant.py` (confirmed: every const-eval weight input is `bf16`, never `float8`). It is **not** the bottleneck.
- The real work is **880 of the 1062 const-evals converting bf16 weights → `bfp_bf8` (block-float8) tiles**, and each one does a **host↔device round-trip** (`to_device → to_layout → from_device → typecast-on-HOST → to_device`). Movement (~2.5× the bf16 weight per weight, serialised over ~1000 calls), not arithmetic, is the dominant cost of the 78.6 min.
- The **352 CPU-hoisted calls are NOT "88×4 fp8 linears"** as the task hypothesised. They are 177 RoPE `inv_freq` reshapes (a 64-float `memcpy`) + 177 fused-weight builds — RoPE/weight-prep, ~2+2 per layer. Small relative to the 880 host round-trips.
- The **bf16→bfp8 host round-trip is deliberate**: tt-mlir commit #8140 (2026-04-29) moved it to host because the **on-device bfp8 typecast kernel is numerically inaccurate** for weights (much higher relative-RMS; host packer restored +48% TOP1 on gpt-oss-120B). A device op *exists* — it was *rejected for accuracy*.
- **TIME and MEMORY are separate problems with different fixes** (see §6). The OOM is **transient conversion inflation**, not steady-state capacity: bfp8 weights (15.2 GiB/chip) + KV (9.56) = 24.8 GiB **fit** under 31.8 GiB/chip, yet DRAM is full at OOM — and KV was never even allocated (no forward pass ran).

---

## 1. fp8→bf16 dequant is host-upfront (CONFIRMED) — it is not the const-eval work

`integrations/vllm_plugin/vllm_tt/fp8_dequant.py:122-189` (`process_weights_after_loading`) runs at **load time on the host**, before any tracing/compile:

- line 176: `deq = (weight.to(torch.float32) * rows).to(target_dtype)` (target = `bf16`), then `replace_parameter(layer, "weight", deq)` (line 189).
- It is installed per-fp8-linear via the monkeypatched `Fp8Config.get_quant_method` (`:221-253`), logging `"using dequantizing linear method"` once per layer.

Confirmed in the FULL log:
- **352** `"using dequantizing linear method"` lines → the 352 fp8 linears are dequantised to bf16 on the host up front.
- **Every** const-eval weight argument is `bf16`, never `float8`: a scan of const-eval signatures gives **882 `bf16` inputs + 176 `f32` inputs, zero fp8**. Representative: `@main_const_eval_1(%arg0: tensor<12288x1536xbf16, ...>)` (log line 144837).

So by the time the graph exists, weights are already bf16. The const-eval/CPU-hoist work operates on bf16 — it is **not** the fp8 dequant.

---

## 2. What the const-evals actually compute

**1062 const-eval funcs, classified by return type (log scan):**

| Return type | Count | What it is |
|---|---:|---|
| `!ttcore.tile<32x32, bfp_bf8>` | **880** | **bf16 weight → bfp8 (block-float8) tile** — the dominant work (10/layer × 88) |
| `f32` (`1x64`) | 176 | RoPE `inv_freq` reshape (2/layer) |
| `bf16` (passthrough) | 2 | `embed_tokens` / `lm_head` full-vocab `to_device` only |

### 2a. The bf16→bfp8 weight conversion (880 funcs) — this is the real work

`main_const_eval_1` (log lines 144837-144849) is the canonical pattern:

```
@main_const_eval_1(%arg0: tensor<12288x1536xbf16, system_memory>)
                 -> tensor<12288x1536x!ttcore.tile<32x32, bfp_bf8>, dram> {
  %1 = ttnn.to_device(%arg0)      // host bf16 (RM)  -> DEVICE bf16
  %2 = ttnn.to_layout(%1)         // DEVICE bf16 RM  -> DEVICE bf16 TILE
  %3 = ttnn.from_device(%2)       // DEVICE bf16 TILE -> HOST bf16 TILE   <-- round-trip back
  %4 = ttnn.typecast(%3)          // HOST bf16 TILE  -> HOST bfp_bf8 TILE (block-float pack)
  %5 = ttnn.to_device(%4)         // HOST bfp8 TILE  -> DEVICE bfp8 TILE
  return %5
}
```

Every weight is moved **host→device→host→device**. Per weight the traffic is roughly *bf16 up + bf16 down + bfp8 up ≈ 2.5× the bf16 weight size*. The bfp8 quantize itself (`ttnn.typecast`) runs **on the host tensor `%3`** — it is a host op, dispatching to tt-metal's block-float packer. `main_const_eval_4/_5/_8` (12288×3584, 3584×12288, 12288×1536) are identical in structure — these are the q/k/v/o + gate/up/down MLP weight shards (TP-sharded: 3584 = 28672/8, etc.).

### 2b. The 352 CPU-hoisted calls are RoPE + fused-weight prep, split evenly 177/177

```
177 × call @cpu_hoisted_const_eval_32f3c969   (64xf32 -> 1x64xf32)
177 × call @cpu_hoisted_const_eval_8aa46810   (3× bf16 -> 12288x1792 bfp8 weight)
```

- `cpu_hoisted_const_eval_32f3c969` (log 192464-192487): body is literally `malloc(64*f32)` + `memcpy` — a **RoPE `inv_freq` reshape** (rank-1 `64` → rank-2 `1x64`). Trivial. Invoked from `main_const_eval_2/_6` (log 144850, 144908).
- `cpu_hoisted_const_eval_8aa46810` (log 192488+): `arg_ranks=[2,2,2] -> result_ranks=[2]`, three f32 inputs (`128×12288`, `128×12288`, `1536×12288`) → `12288×1792` f32. A CPU-side **fused-weight build** (transpose/pad/concat producing the 12288×1792 gate/up shard). Invoked inside `main_const_eval_3/_7` (log 144859-144881), which then `typecast → bf16 → to_device → from_device → typecast → bfp8 → to_device` (same host round-trip appended).

**This overturns the task's assumption that "352 = 88×4 fp8 linears."** The 352 hoist-calls are ~2 RoPE + ~2 fused-weight-build per layer (177+177 ≈ 2×88). The fp8-linear count (352) is a coincidence — those live as the 352 `"dequantizing linear method"` host-dequant events in §1, not as CPU-hoist calls.

---

## 3. Why bf16→bfp8 is host-side (and why the round-trip)

**A device bfp8 typecast DOES exist**, but it was **deliberately abandoned for weights on accuracy grounds.**

- Device path exists: tt-metal `unary.cpp:32-33` sets `bfp8_pack_precise` when `output_dtype == BFLOAT8_B`; the TTNN IR op carries a `bfp8_pack_precise` attribute (`TTNNOpsAttrs.td:1115,1129`). `TTNNDecomposeLayouts.cpp:512-521` even documents that a **TILE→TILE typecast is "always-supported" on device**. So mechanically, the `%3→%4` step in §2a *could* run on-device.

- It is forced to host by the weight-dtype pass, unconditionally for BFP formats:
  `lib/Dialect/TTNN/Transforms/TTNNWeightDtypeConversion.cpp:102-127` hard-codes `from_device → typecast(host) → to_device` for `BFP_BFloat4/BFP_BFloat8`; comment (`:102-106`): *"The host typecast dispatches to tt-metal's host packer for BFP formats. Const-eval results are cached at compile time, so the host roundtrip is paid once per cached weight per program."* Non-blockfloat (bf16) targets keep the **single device typecast** (`:128-133`).

- **The reason is numerical, and documented.** Commit `3e4cfe5daa` "Weight dtype casting using typecast on host (#8140)" (2026-04-29):
  > *"accuracy issues on GPT OSS 120B were caused by on-device ttnn::typecast kernel producing much higher relative-RMS than tt-metal's host packer on the same bf16 input. Typecast on host restores accuracy."*
  Measured impact: `gpt_oss_120b` TOP1 39.06% → 87.50% (**+48.44%**) by moving the bfp8 pack to the host packer.

- **Is the round-trip data movement the dominant cost?** Yes, almost certainly. Per §2a the arithmetic (a block-float pack of an ~88 MB tensor) is fast; the cost is the **per-weight host↔device marshaling** (bf16 up, bf16 down, bfp8 up), serialised across ~1000 const-eval calls on a 32-chip mesh with per-call dispatch overhead. 4718 s / 1062 calls ≈ 4.4 s/call average — consistent with marshaling + dispatch, not with pack arithmetic.

### `isOnDeviceLayoutChangeSupportedForDataType` (`TTNNDecomposeLayouts.cpp:110-117`)
Returns true only for `bf16/f32/uint32/uint16/int32` — **not bfp8**. So a bfp8 tensor cannot be (un)tilized on device; the block-float *tilize* is inherently a host-packer operation. This reinforces that "keep it all on device" is not a trivial toggle — even if the typecast ran on device, bfp8 layout ops don't.

---

## 4. Where the ~79 min goes (reasoned from structure)

Wall-clock anchors (loguru timeline):

| Event | Timestamp | Elapsed |
|---|---|---:|
| Process start | 19:37:33 | 0 s |
| Device/fabric init done, last MLIR module dumped (`ttnn_runtime`) + final fabric override | 19:43:55 | **337.6 s** |
| **TT_FATAL DRAM OOM** | 21:02:34 | **5056.2 s** |

**Silent window = 5056.2 − 337.6 = 4718.6 s ≈ 78.6 min**, entirely **post-compile runtime**, with **0 `Executing operation` lines** — the run **died in const-eval weight materialisation before any forward op.**

Best-supported attribution: the 78.6 min is the runtime executing the **1062 `load_cached` const-eval functions** (IR contains exactly 1062 `load_cached`), overwhelmingly the **880 bf16→bfp8 host round-trips**, until DRAM overflows partway through. The cost is **per-call host↔device marshaling across ~1000 serialised calls**, not the pack arithmetic and not the tilize.

*Caveat on the silence (stated, not hand-waved):* the sibling 2-layer run `devstral_dptp_test_synced_cpu_false.log` logged **703 `Executing operation` + 148 `load_cached`** lines, so const-eval/device ops *are* loggable — the FULL run simply wasn't in per-op logging mode, so the silence is a **verbosity difference, not evidence of host-boundness**. The timing attribution does not rest on the silence; it rests on (a) the fixed 337.6 s→5056.2 s runtime window, (b) 0 forward ops (died in const-eval), (c) the 1062 `load_cached` + 880 host round-trips in the IR. **What would confirm it directly:** per-op device profiling, or the `Cache miss/hit` timestamps enabled during this window.

---

## 5. Fix methodologies for the REAL bottleneck (bf16→bfp8 / const-eval)

### (a) Device op — bf16→bfp8 on device so the compiler doesn't host-round-trip

- **The op already exists** (`unary.cpp:32`, `bfp8_pack_precise`; TILE→TILE typecast is device-supported). So this is **not a new-kernel effort** — it's a tt-mlir change to `TTNNWeightDtypeConversion.cpp:111-127` to emit a device typecast instead of `from_device→typecast→to_device`.
- **BUT it reintroduces a known, measured correctness regression.** #8140 moved it to host *precisely because* the device bfp8 packer's relative-RMS is much worse (−48% TOP1 on gpt-oss-120B). Flipping it back trades 79 min for an accuracy cliff. The genuinely *hard* work the user anticipated is **making the device bfp8 packer numerically accurate** (a tt-metal kernel problem), not writing a packer at all.
- Verdict: **not safe as-is.** Viable only if paired with fixing the device packer's precision (`bfp8_pack_precise` path) and validating PCC — real tt-metal kernel effort.

### (b) Host-upfront — precompute the final on-device representation in the weight loader

- **bfp8 is not representable as a host torch tensor.** Block-float8 is a Tenstorrent tiled format (shared exponent per block within a 32×32 tile); torch has no such dtype, and there is no on-device (un)tilize for bfp8 (`TTNNDecomposeLayouts.cpp:110-117`). So the fp8_dequant-style host handoff **can only produce bf16** — which weights *already are*. The bfp8 pack must still happen somewhere (host packer or device).
- The only way to "precompute bfp8" on host is to reimplement tt-metal's exact block-float tile byte-layout in the loader and hand it off as an opaque `uint8` buffer tagged bfp8. Fragile (must byte-match the packer), no torch dtype, easy to get wrong. **Not recommended.**
- Verdict: **cannot eliminate the bfp8 conversion.** Host-upfront already goes as far as it can (bf16). No leverage on this bottleneck.

### (c) Skip bfp8 — run bf16 on-device weights

- Mechanically clean: if `experimental_weight_dtype` is not bfp8 and weights are already bf16, the weight-dtype pass **no-ops** (`TTNNWeightDtypeConversion.cpp:90-95`) → the **880 host round-trips disappear entirely.** (The ~176 RoPE + ~177 fused-weight const-evals remain, now bf16-output and cheaper — so this removes the round-trips, not "all const-eval.") **This fixes the TIME problem.**
- **But it makes the MEMORY problem worse and un-runnable:** bf16 weights are **28.6 GiB/chip** vs bfp8 **15.2 GiB/chip**; + 9.56 GiB KV = **38.2 GiB > 31.8 GiB DRAM/chip** → guaranteed OOM even at steady state. bfp8 is *required* to fit.
- Verdict: **fixes time, breaks memory.** Only usable if paired with more chips / higher TP so bf16 fits, or as a diagnostic to prove the round-trip is the time cost.

---

## 6. Recommendation — separate the TIME problem from the MEMORY problem

They have different root causes and different best fixes.

### The MEMORY (OOM) problem is transient conversion inflation, not capacity — fix it first

DRAM arithmetic (per chip; DP replicates, TP=8):

| Item | GiB/chip |
|---|---:|
| DRAM available | **31.8** (8 banks × 3.98; OOM line confirms bank 4.269/4.272 GiB = **full**) |
| bfp8 weights (steady state) | 15.2 |
| KV budget (gpu_util 0.30) | 9.56 |
| **bfp8 weights + KV** | **24.8** ← fits, ~7 GiB headroom |
| bf16 weights + KV | 38.2 ← does not fit |

Two facts prove the OOM is **transient**, not steady-state: (1) steady-state bfp8+KV (24.8) fits under 31.8 with headroom, yet DRAM is *full* at OOM; (2) **KV was never allocated** — KV sizing runs during forward-pass profiling, and **0 forward ops ran**. So the 31.8 GiB was consumed by **weights + const-eval transients alone.** The transients are exactly the §2a round-trip staging: each conversion holds a **device bf16 copy** (the `to_device` before `from_device`) and, on the 177 `8aa46810` path, **f32 intermediates** (`typecast bf16→f32`, doubling again), on top of the resident bfp8 outputs. The failing allocation is a modest **84 MiB (88,080,384 B)** buffer landing on an already-full DRAM.

**Best memory fixes (in order):**
1. **Bound const-eval residency / free transients eagerly** so peak = resident bfp8 + one weight's transient, not resident + many bf16/f32 stagings. This is the most direct fix and keeps bfp8 (required to fit). Investigate whether const-eval materialisation can stream (materialise → offload → free) rather than hold all inputs+outputs.
2. **Raise gpu-memory headroom during load** — the OOM is before KV alloc, so the const-eval phase needs the transient headroom; lowering concurrent transients matters more than the 0.30 util.
3. If transients can't be bounded, **increase TP or chip count** to shrink per-chip weight footprint.

### The TIME (78.6 min) problem — the 880 host round-trips

- The wise path is **not** fix (a) as-is (accuracy regression) nor (c) as-is (breaks memory). It is to **attack the movement, not the dtype**:
  - **Bound/parallelise the const-eval marshaling** — the same eager-free/streaming change that fixes memory also removes redundant bf16 device staging, cutting the per-weight traffic from ~2.5× toward ~1× (host bf16 → host bfp8 pack → device bfp8, skipping the device→host bf16 bounce). The `to_device→to_layout→from_device` prefix in §2a is the avoidable part; the host pack + final `to_device` is irreducible while accuracy requires the host packer.
  - **Longer-term / higher-effort:** fund the tt-metal device bfp8-packer precision work so fix (a) becomes safe — then the whole conversion stays on-device and both problems vanish. This is the "hard kernel" the user foresaw; it's hard because of *accuracy*, not because the op is missing.

### One-line recommendation
Keep bfp8 (bf16 doesn't fit). The OOM is **transient const-eval inflation** — fix it by bounding const-eval residency / freeing the bf16+f32 stagings eagerly, which *also* cuts the 78.6-min round-trip traffic. Do **not** flip bfp8→device typecast (`#8140` accuracy regression) and do **not** switch weights to bf16 (guaranteed OOM at 88 layers). Making the device bfp8 packer accurate is the only route to eliminating the conversion entirely, and that is a tt-metal precision effort, not a tt-mlir toggle.

---

### Key citations
- `integrations/vllm_plugin/vllm_tt/fp8_dequant.py:122-189` — host-upfront fp8→bf16 dequant (`:176` the multiply, `:189` replace_parameter); `:221-253` per-linear install.
- FULL log `144832-144951` — const-eval bodies (`_0` embedding passthrough; `_1/_4/_5/_8` the bf16→bfp8 host round-trip; `_2/_3/_6/_7` the CPU-hoist callers).
- FULL log `192464-192487` (`32f3c969` = 64-float memcpy / RoPE reshape), `192488+` (`8aa46810` = fused-weight build).
- FULL log counts: 1062 `main_const_eval`, 1062 `load_cached`, 880 bfp8 returns, 352 `cpu_hoist_call` (177+177), 352 `"dequantizing linear method"`, 0 `Executing operation`.
- FULL log timeline: compile end `337.587s` (line 197262), OOM `5056.202s` (197264-197268), buffer 88,080,384 B on full DRAM (bank 4,269,179,520 / 4,272,341,376 B).
- `third_party/tt-mlir/.../TTNNWeightDtypeConversion.cpp:102-133` — hard-coded host round-trip for BFP; bf16 single device typecast.
- `third_party/tt-mlir/.../TTNNDecomposeLayouts.cpp:110-117` (bfp8 not device-(un)tilizable), `:498-521` (`layoutChangeNeedsHost` / `canKeepTypecastOnDevice`, TILE→TILE device typecast).
- tt-metal `.../eltwise/unary/unary.cpp:32-33` + `TTNNOpsAttrs.td:1115,1129` — device bfp8 typecast (`bfp8_pack_precise`) exists.
- tt-mlir commit `3e4cfe5daa` (#8140, 2026-04-29) — device bfp8 typecast abandoned for weights due to relative-RMS accuracy (gpt-oss-120B +48% TOP1 on host packer).
- Sibling `devstral_dptp_test_synced_cpu_false.log` — 703 `Executing operation` + 148 `load_cached` (proves silence is verbosity, not host-boundness).
