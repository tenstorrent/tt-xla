# FLUX.1-dev Transformer OOM — Session Handoff (TEMP COMMIT)

> **This file + its commit are temporary scratch for continuing in a fresh chat.**
> Remove when done: `git reset --soft HEAD~1` (un-commit, keep files) or
> `git reset --hard HEAD~1` (discard). This commit is **local only — not pushed.**

Date: 2026-06-26 · Host: `-proj-sw-user-dev-ctr-akannan-26-jun-yyz` · Device: **single** Blackhole p150 (`/dev/tenstorrent/0`)

---

## Goal
Test the FLUX.1-dev component tests on **latest tt-xla main**, confirm an OOM regression
seen earlier, comment out the T5 xfail to read true status, and decide single-chip vs multichip.

Related PRs (both OPEN, branch `akannan/bringup_flux1`):
- tt-xla **#5308** — Flux1-dev component tests (`tests/torch/models/flux/`)
- tt-forge-models **#773** — Flux1-dev component loaders (`flux/pytorch/`)

## Environment / setup state
- tt-xla HEAD: **`4bcaf64ae`** (tt-mlir uplifted to `aae51da9...`). Plugin `.so` rebuilt 06:28 today → reflects this main.
- Overlaid onto main from the PR branches (NOT committed to those PRs):
  - tt-xla working tree: `tests/torch/models/flux/{__init__,test_clip_text_encoder,test_t5_text_encoder,test_transformer,test_vae_decoder}.py`
  - submodule `third_party/tt_forge_models` (pin `8ac99db`, files checked out from `origin/akannan/bringup_flux1`):
    `flux/pytorch/loader.py` (M), `flux/pytorch/src/__init__.py` (A), `flux/pytorch/src/model_utils.py` (A)
- **T5 xfail is currently COMMENTED OUT** in `test_t5_text_encoder.py` (to read true status). Needs restoring before any PR push.
- HF: FLUX.1-dev is cached in `HF_HOME=/proj_sw/user_dev/ctr-akannan/.cache/huggingface`. Gated repo — needs a token (`HF_TOKEN`) exported when running.
- Run logs from this session: `/tmp/flux_results/*.log` (may be cleared on reboot).

## Results — single-chip Blackhole p150, bf16 (latest main)

| Component | Result | Notes |
|---|---|---|
| CLIP text encoder | ✅ PASS | 339s |
| VAE decoder | ✅ PASS | 540s |
| T5 text encoder | ❌ FAIL | PCC **0.9575** < 0.99 — known precision-drift **#5250**, NOT OOM. xfail is legitimate. |
| Transformer (`dram_space_saving`) | ❌ FAIL | **DRAM OOM regression** (was PASS @ PCC ~0.9998 on earlier commits) |

### The OOM (transformer baseline)
Runtime DRAM alloc failure (compile OK; fails in `ExecuteComputation` → `BankManager::allocate_buffer`):
```
Out of Memory: allocate 132120576 B DRAM buffer across 7 banks (18874368 B/bank),
bank size 4272341376 B (allocated 4271876224, free 465152, largest free 219392)
```
- Total usable DRAM = 7 × 4,272,341,376 B = **27.85 GiB**; device **99.99% full**, short by **122.9 MiB (0.43%)**.
- Failing buffer = 132,120,576 B = **66 M bf16 elements** (one intermediate).
- Surfaces as `RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13`.
- It PASSED on earlier commits → this is a **regression** from the tt-mlir/tt-metal uplift growing the footprint past the edge.

## Single-chip fit levers — ALL EXHAUSTED, none helped
(experiment test files live in `tests/torch/models/flux/test_transformer_fit*.py`)

| Lever (file) | Result |
|---|---|
| baseline `dram_space_saving` | OOM, short 122.9 MiB |
| `experimental_weight_dtype="bfp_bf8"` (`fitB_bfp8`) | **SIGSEGV** |
| `bfp_bf8` w/o dram_space_saving (`fitB_bfp8_nodram`) | **SIGSEGV** (identical → dram-independent) |
| `optimization_level=2` (`fitA_opt2`) | **HANG** in compile, >20 min (killed) |
| `fp32_dest_acc_en=False` (`fitC_nofp32acc`) | **identical OOM** (zero DRAM change) |

### bfp8 SIGSEGV — root cause (gdb backtrace captured)
Runtime crash in **tt-metal host BFP8 packer**, via the const-eval weight-prep typecast — NOT a compile pass, NOT dram_space_saving:
```
#0 pack_as_bfp_tiles<(tt::DataFormat)6=Bfp8_b, float>(span<float const>...)   ← libtt_metal.so  ★
#1 pack_as_bfp8_tiles<float>
#2 transform_buffers<float, bfloat8_tag>
#4 tt_metal::to_dtype(HostTensor, DataType)
#6 ttnn::to_dtype  #8 ttnn::typecast
#9 runtime::ttnn::operations::layout::run(TypecastOp)
#12 cache::run(LoadCachedOp)   ← const-eval weight pre-pack
```
Almost certainly an OOB read packing a weight buffer into BFP8 16-elem-block/32×32-tile layout.
**Fix is upstream (tt-metal / tt-mlir runtime).** Even if fixed, bfp8 is global-only & precision-risky (T5 already drifts in plain bf16).

## Hardware reality
- **This host has only ONE TT device** → multichip **cannot be RUN here**.
- Multichip sizing for the **~24 GB bf16** (12B-param) transformer, usable ≈ 87% of physical:
  - **n300 (2× Wormhole n150, ~10.4 GB usable/chip): ❌ does NOT fit** — 24/2 = 12 GB weights/chip > 10.4, before activations.
  - **4× Wormhole (LoudBox/quietbox): ✅** ~6 GB weights/chip, ~4 GB for activations. (Minimum viable Wormhole.)
  - **8× Wormhole (T3000): ✅** comfortable.
  - **2× Blackhole: ✅** ~12 GB weights/chip + ~16 GB headroom.
- User was going to check n300 → **answer: n300 insufficient; reserve 4× Wormhole or 2× Blackhole.**

## PLANNED NEXT STEPS (pick up here in new chat)
1. **Decide/secure target HW**: 4× Wormhole (LoudBox) or 2× Blackhole. (n300 ruled out.)
2. **Design tensor-parallel sharding** for FluxTransformer2DModel (Megatron-style column→row on the
   linears; shard attention heads; minimize CCLs). Consider the `sharding-model-analysis` skill.
3. **Write the multi-device transformer test scaffold** (Shardy annotations) — can be authored on this
   host; runs only on the multi-chip box.
4. **(Parallel, optional) File upstream issues:**
   - tt-metal: SIGSEGV in `pack_as_bfp_tiles<Bfp8_b,float>` packing FLUX transformer weights to bfp8_b.
   - tt-mlir/tt-metal: transformer DRAM OOM **regression** (passed on earlier commits) — consider bisecting
     the tt-mlir pin between the PR's working commit and `aae51da9` (requires plugin rebuilds).
5. **Before pushing PR #5308**: restore the T5 xfail; delete the temp `test_transformer_fit*.py` experiment
   files and this handoff file + its commit.

## Cleanup checklist for this temp work
- `git reset --soft HEAD~1` to drop this commit.
- `rm FLUX1_TRANSFORMER_OOM_HANDOFF.md tests/torch/models/flux/test_transformer_fit*.py`
- Restore the xfail in `tests/torch/models/flux/test_t5_text_encoder.py` (uncomment the `@pytest.mark.xfail(...)`).
