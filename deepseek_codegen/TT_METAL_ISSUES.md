# tt-metal kernel issues blocking DeepSeek-V3.2 decode fusions

**Status: drafts, not filed yet.** Backing notes: `deepseek_codegen/FUSION_NOTES.md`.

Scope: each section below is one filing candidate. Three are runtime kernel bugs; four are Blackhole / 4×8-mesh feature gaps in deepseek-named ops. The codegen-emitter gaps (already split out under `TT_MLIR_PARENT.md` / `TT_MLIR_SUB1-4_*.md`) are deliberately out of scope here.

`## Repro` sections are placeholders — to be added later.

## Pre-filing triage vs existing tt-metal issues

Searched `tenstorrent/tt-metal` (authenticated GitHub search) for each draft. Summary:

| draft | overlap status | relevant existing issue(s) | recommended action |
| --- | --- | --- | --- |
| #1 sparse_matmul `out_subblock_w>1` hang | **novel** | none | file as new |
| #2 `rms_norm_post_all_gather` slice rank | **already covered** | [#43962](https://github.com/tenstorrent/tt-metal/issues/43962) *(in-flight PR: "ttnn::slice: accept slice params with fewer dims than input rank")* | do **not** file new — comment on #43962 noting `rms_norm_post_all_gather` is a concrete blocked caller |
| #3 `rms_norm_post_all_gather` NaN/Inf on BF16 stats | **partial / novel** | [#43527](https://github.com/tenstorrent/tt-metal/issues/43527) *(closed: low PCC on `(1,1,32,2**x)`)*, [#43807](https://github.com/tenstorrent/tt-metal/issues/43807) *(closed: garbage with `use_2d_core_grid=True`)*, [#39040](https://github.com/tenstorrent/tt-metal/issues/39040) *(open: HiFi4 + FP32 accum precision work)* — all different code paths | file as new; cross-reference the related precision/correctness work |
| #4 parent: deepseek ops on 4×8 BH | **already covered** | [#43444](https://github.com/tenstorrent/tt-metal/issues/43444) *(open, assigned to amorrisonTT + dchenTT: "MoE: Full fused op on BlackHole")* + [#43944](https://github.com/tenstorrent/tt-metal/issues/43944) *(open: "4×8 Sub-torus bring-up/validation")* | do **not** file new parent — consolidate sub-drafts under #43444 |
| #5 `moe_gate_mm` BH program factory | **novel beyond closed precedent** | [#41405](https://github.com/tenstorrent/tt-metal/issues/41405) *(closed: "moe_gate_mm: replace UB with TT_FATAL for non-Wormhole architectures")* — the TT_FATAL we hit was deliberately added there | file as new — the TT_FATAL closes the UB gap but doesn't add BH support |
| #6 `deepseek_moe_reduce_scatter` BH | **novel** | none directly; [#42697](https://github.com/tenstorrent/tt-metal/issues/42697) *(open: "Check functionality with non 32 UPR")* is the closest in-flight work | file as new |
| #7 `moe_compute` / `moe_gpt` 4×8 mesh + `experts_per_device=8` | **already covered** | [#43444](https://github.com/tenstorrent/tt-metal/issues/43444) *(BH support, open)*; the `experts_per_device` axis already landed via [#41797](https://github.com/tenstorrent/tt-metal/issues/41797) *(closed PR removing the 3-experts/device cap by moving the active-tokens semaphore to L1)* | do **not** file new — comment on #43444 with our 4×8 + `experts_per_device=8` use case |
| #8 `selective_reduce_combine` 4×8 mesh | **already covered** | [#43444](https://github.com/tenstorrent/tt-metal/issues/43444) (the fused MoE BH umbrella includes the selective_reduce_combine path) | do **not** file new — same comment thread as #7 |

**Net filing list after triage:** 4 new issues (#1, #3, #5, #6) + 1 follow-up comment on **#43962** + 1 consolidated comment on **#43444** covering drafts #4 / #7 / #8.

## Uplift status of referenced PRs against our tt-metal pin

Our current pin (`mvasiljevic/deepseek-router-fuse` HEAD → tt-mlir `297f7eb6c` → tt-metal **`c5ebc6351`**, dated 2026-05-19 06:39 UTC). Ancestry verified via `git merge-base --is-ancestor`.

| PR | merged | in our pin? | what it does | effect on our drafts |
| --- | --- | --- | --- | --- |
| [#41405](https://github.com/tenstorrent/tt-metal/pull/41405) | 2026-04-06 | **IN** | `moe_gate_mm`: replace UB with TT_FATAL for non-WH architectures | This is exactly the TT_FATAL we hit (draft #5). Our blocker is real on current pin. |
| [#41797](https://github.com/tenstorrent/tt-metal/pull/41797) | 2026-04-14 | **IN** | MoE: arbitrary experts per device (moved active-tokens semaphore from a 32-bit field to a sharded L1 tensor, removing the 3-experts/device cap) | The `experts_per_device > 3` axis of draft #7 is already unblocked on the WH paths — only the BH program factory side is still missing. |
| [#43818](https://github.com/tenstorrent/tt-metal/pull/43818) | 2026-05-07 | **IN** | fix `rms_norm_pre/post_all_gather` garbage output with `use_2d_core_grid=True` (closes #43807) | Different code path from our draft #3 (we don't use `use_2d_core_grid=True`). Our NaN/Inf remains reproducible on current pin. |
| [#43602](https://github.com/tenstorrent/tt-metal/pull/43602) | 2026-05-08 | **IN** | `rms_allgather`: post-reduce data-format-reconfig fix (closes #43527 low PCC on `(1,1,32,2**x)`) | Different shape regime from our draft #3 — we hit Inf, not low-PCC. Our case is still reproducible. |
| [#43932](https://github.com/tenstorrent/tt-metal/pull/43932) | 2026-05-18 | **IN** (barely — 1 day before pin) | Generalize `moe_compute` shape support beyond hardcoded model configs | Removes one axis of the "topology-incompatible" blocker on draft #7. BH program factory still missing. |
| [#39044](https://github.com/tenstorrent/tt-metal/pull/39044) | 2026-03-04 | **IN** | Distributed RMSNorm: allow `fp32_dst_acc_en` to be set in pre and post | Mentioned for context — gives us the precision knob if draft #3 turns out to be FP32-vs-BF16-accum related. |
| [#30637](https://github.com/tenstorrent/tt-metal/pull/30637) | 2025-10-28 | **IN** | Improve `rmsnorm_distributed` correctness with FP32 reductions + more accurate `rsqrt` | Mentioned for context. |
| [#44195](https://github.com/tenstorrent/tt-metal/pull/44195) | 2026-05-25 | **OUT** (6 days after pin) | MoE compute: Blackhole single-card bring-up | First BH MoE bring-up; will arrive in the next tt-mlir uplift past our current pin. Tracks against [#43444](https://github.com/tenstorrent/tt-metal/issues/43444) parent. |
| [#43962](https://github.com/tenstorrent/tt-metal/pull/43962) | **draft, not merged** (last update 2026-05-10, 1 commit) | n/a | `ttnn::slice`: accept slice params with fewer dims than input rank | Draft #2 stays blocked until this lands. No ETA visible; PR is currently `draft=true` with `mergeable_state=unknown`. |

**Practical reading.** Three closed RMSNorm fixes (#43818, #43602, #43932) are already in our pin and do **not** address our draft #3 — the NaN/Inf failure mode in E2 is on a different code path and stays reproducible. One in-flight PR (#43962) covers our draft #2 but is still in draft state. One BH bring-up PR (#44195) just-merged after our pin will partially unblock the parent BH MoE umbrella (#43444) at the next tt-mlir uplift.

Suggested filing order (best repro-cost-to-impact ratio first):

1. [bug] `sparse_matmul` hangs with `out_subblock_w > 1`
2. [bug] `rms_norm_post_all_gather` slice rank assertion
3. [bug] `rms_norm_post_all_gather` NaN/Inf on BF16 stats
4. [feature] deepseek-named ops — 4×8 Blackhole + `experts_per_device=8` support (parent for #5–#8)
5. [feature] `experimental.deepseek.moe.moe_gate_mm` BH program factory
6. [feature] `experimental.deepseek_moe_reduce_scatter` BH program factory
7. [feature] `experimental.moe_compute` / `moe_gpt` 4×8 mesh + `experts_per_device=8`
8. [feature] `experimental.ccl.moe.selective_reduce_combine` 4×8 mesh support

---

## 1. [bug] `sparse_matmul` hangs with `out_subblock_w > 1` (no validation, no TT_FATAL)

**Pre-filing check:** novel — no existing tt-metal issue covers this. **File as new.**

**Severity:** high — silent hang requires `kill -KILL` + `tt-smi -glx_reset_auto`. `tt-perf-report` actively recommends `out_subblock_w ≥ 2` via the SLOW/"Output subblock 1x1 is small" hint, making this a foot-gun for anyone tuning sparse_matmul.

**Context.** Two attempts to follow the perf-report hint on the DeepSeek-V3.2 MoE sparse_matmuls both hung the device. The hang reproduces with multi-tile pack (multi-sub-block reduce) AND without — ruling out the "multi-sub-block reduce unsupported" hypothesis the first attempt suggested.

**Evidence.**
- Attempt A: `out_subblock_w=1 → 2` on **both** sparse_matmuls at `per_core_N=4`. Wedged 44 minutes at 746 % CPU before kill.
- Attempt B: `out_subblock_w=1 → 2` on a single sparse_matmul at `per_core_N=2` — exactly 1 output sub-block per core, no multi-sub-block reduce. Same hang signature (zero stdout in 4-minute window).
- Surrounding config (both attempts): `MatmulMultiCoreReuseMultiCast1DProgramConfig`, `in0_block_w=4`, sparse mask via `ttnn.sparse_matmul`.

**Proposed fix.** Two acceptable resolutions:
1. **Support `out_subblock_w > 1`** in the sparse-mask reader path (the sparse-aware reader appears to assume a 1-tile-wide pack pattern; multi-tile pack would need the sparse-mask offsets recalculated per sub-tile).
2. **Reject the configuration in validation** with a TT_FATAL listing the supported `out_subblock_w` range for `ttnn.sparse_matmul`, so users following the perf-report hint get an immediate clear error instead of a silent device deadlock.

Either is acceptable; the silent hang is the worst failure mode.

**Repro.** TBD.

---

## 2. [bug] `rms_norm_post_all_gather` (and `wan_fused_rmsnorm_post_allgather`) trip `slice.cpp:35` rank assertion on rank-4 inputs

**Pre-filing check:** **already covered by [#43962](https://github.com/tenstorrent/tt-metal/issues/43962)** ("ttnn::slice: accept slice params with fewer dims than input rank", open PR, branch `devisettym/slice_op`). **Do not file a new issue.** Add a comment on #43962 naming `rms_norm_post_all_gather` / `wan_fused_rmsnorm_post_allgather` as concrete callers blocked by the assertion, with the input shape evidence below.

**Severity:** medium — blocks the canonical fused-distributed-RMSNorm path for any model whose RMSNorm carries a rank-4 `[1, 1, S, H]` activation tensor (which is the natural shape after `reshape` for tile-padded distributed normalization).

**Context.** DeepSeek-V3.2 decode has 5 sites of manual distributed RMSNorm. The canonical replacement is `ttnn.rms_norm_post_all_gather` (regular) or `ttnn.experimental.wan_fused_rmsnorm_post_allgather` (with a `num_heads_per_device` knob). Both abort at the same internal slice with `TT_FATAL @ slice.cpp:35: input_rank == begins.size() -- Input rank 4 and begins 2 must have the same size`. The slice with `begins=[0,0]` is inside the op's internal stats-extraction path — no Python frame in the trace beyond the post-allgather call itself.

**Evidence.**
- TT_FATAL message: `ttnn/cpp/ttnn/operations/data_movement/slice/slice.cpp:35`. Exact assertion: `TT_FATAL(input_rank == begins.size(), "Input rank {} and begins {} must have the same size", input_rank, begins.size());`.
- Dispatch chain: `ttnn::rms_norm_post_all_gather` (`normalization/rmsnorm_distributed/rmsnorm_post_all_gather.cpp:42-55`) → `ttnn::prim::layer_norm_post_all_gather` → internal stats-extraction `slice(stats, begins=[0,0], …)` regardless of input rank.
- Reproduced against both the regular and WAN variants. The WAN variant inherits the same internal slice kernel, so this is **one fix for both ops**.
- Tested input shape: `input=[1, 1, 32, 896]`, `gathered_stats=[1, 1, 32, 256]` (32 tile-aligned cols × 8 devices on cluster_axis=1).

**Proposed fix.**
1. Make the internal stats-extraction slice rank-aware — pad `begins` to `input_rank` with leading zeros (or generate the slice with the correct rank from the op's program factory).
2. Alternatively, reshape the stats to rank-2 `[S, H]` inside the op before the slice, then reshape back. The op also internally slices stats by hidden_size so the reshape needs to thread through both stages.

**Repro.** TBD.

---

## 3. [bug] `rms_norm_post_all_gather` produces NaN/Inf when validation passes (BF16 stats + `all_gather(dim=3, cluster_axis=1)`)

**Pre-filing check:** novel for this configuration. Related but distinct: [#43807](https://github.com/tenstorrent/tt-metal/issues/43807) (closed; garbage output with `use_2d_core_grid=True` — different code path), [#43527](https://github.com/tenstorrent/tt-metal/issues/43527) (closed; low PCC on `(1,1,32,2**x)` 1×2 mesh — different shape regime), [#39040](https://github.com/tenstorrent/tt-metal/issues/39040) (open; "Deepseek: Switch to HiFi4 with FP32 accumulation for RMSNorm" — related precision work, likely worth cross-referencing). **File as new**, cross-reference all three.

**Severity:** medium — validation accepts a configuration the kernel cannot execute correctly. Silent numeric failure is worse than a TT_FATAL: PCC harnesses with `nan_to_num`-style preprocessing can read PCC=1.0 false positives when both candidate and golden contain `inf`.

**Context.** Attempted the full RMSNorm-post-allgather fusion on top of an already-working pre-allgather fusion: `pre op dtype=BFLOAT16`, `all_gather dim=3 cluster_axis=1` (gather stats along hidden dim instead of stacking along a new leading dim), then one `rms_norm_post_all_gather` replacing the entire manual ε-add / rsqrt / mul / γ tail. Validation passes — the call shape matches the upstream reference test (`models/.../test_distributed_rmsnorm_allgather.py`). But the per-op output contains NaN/Inf: KV cache `max|Δ|` reaches 2.2e38; sampled tokens 100 % diverge.

**Evidence.** Three open root-cause hypotheses (none confirmed; needs a `ttnn.to_torch` dump at one site before retry):
1. The pre op's tile-wide stats output is a `[..., 1, 32]` tensor where only column 0 holds the real `Σx²`; columns 1-31 are undefined. The post op's row-reduce may average across all 32 cells per row, pulling in garbage. (Kernel may rely on a specific scaler-tile pattern that isn't being fed correctly under HiFi4 + `fp32_dest_acc_en=True` + a 1D core grid + `H=32` rows.)
2. Gamma in our codegen is `[1, 896]` **FP32 in TILE layout**. Validation accepts it (no dtype check for TILE-layout gamma), but the kernel may silently fall back to a `Float16_b` path that misinterprets FP32 tiles. The reference test uses BF16 gamma.
3. `all_gather(dim=3, cluster_axis=1)` may produce a different layout than the reference test's `ttnn.concat(stats, dim=3)` simulation. Worth checking against an actual multi-device unit test.

**Proposed fix.** Investigate the three hypotheses above and either (a) fix the kernel to handle the failing configuration, or (b) tighten validation to TT_FATAL with a clear message naming the unsupported combination (e.g. "FP32 gamma in TILE layout unsupported; pass BF16 gamma" if hypothesis 2 is the root cause).

**Repro.** TBD. Note: depends on issue #2 (slice rank assertion) being fixed first — otherwise the slice TT_FATAL short-circuits the kernel before the numeric failure manifests.

---

## 4. [feature] Parent: deepseek-named ops — Blackhole 4×8 mesh + `experts_per_device=8` support

**Pre-filing check:** **already covered.** The umbrella for "MoE fused ops on BlackHole" is [#43444](https://github.com/tenstorrent/tt-metal/issues/43444) (open, assigned to amorrisonTT + dchenTT). The 4×8 mesh topology tracker is [#43944](https://github.com/tenstorrent/tt-metal/issues/43944) (open). The `experts_per_device > 3` cap was removed in [#41797](https://github.com/tenstorrent/tt-metal/issues/41797) (closed PR — moved active-tokens semaphore to L1). **Do not file a new parent.** Sub-drafts #5/#6/#7/#8 below should be consolidated into a single comment thread on #43444 instead.

**Severity:** blocks every deepseek-specific fused MoE op on our target hardware (Blackhole galaxy `g08blx03`, 32 chips, mesh shape 4×8).

**Context.** The catalogue of `experimental/deepseek*` ops in tt-metal is wormhole-TG-tuned. None currently support the 4×8 BH topology with `experts_per_device=8` (256 experts / 32 devices). The four child issues (#5–#8) each name a specific op; this parent is the umbrella for the topology gap.

**Evidence.** Survey results in `FUSION_NOTES.md` "Wanted but couldn't land" section. Each child issue cites its own test gating or `TT_FATAL`.

**Proposed fix.** For each child op below, add a Blackhole program factory and extend test parameterisation to include `mesh_shape=(4,8)` and (where the op has the concept) `experts_per_device ∈ {4, 8}`. Track #5–#8 against this parent.

---

## 5. [feature] `experimental.deepseek.moe.moe_gate_mm` — add Blackhole program factory

**Pre-filing check:** novel beyond closed precedent. [#41405](https://github.com/tenstorrent/tt-metal/issues/41405) (closed: "moe_gate_mm: replace UB with TT_FATAL for non-Wormhole architectures") is the issue under which the WH-only TT_FATAL we hit was deliberately added — it closes the UB gap but explicitly does not add BH support. **File as new**, naming #41405 as the precedent.

**Context.** Kernel hard-fails on any non-Wormhole architecture. Our 4×8 BH galaxy has a different DRAM-bank-to-core mapping (not 12 banks aligned to a fixed ring).

**Evidence.**
- `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/moe_gate_mm/device/moe_gate_mm_program_factory.cpp:30-37`:
  ```cpp
  constexpr uint32_t required_cores = 12;
  TT_FATAL(num_cores == required_cores,
      "moe_gate_mm requires exactly {} DRAM-aligned cores (Wormhole); got {}. "
      "This op's ring algorithm is hardcoded for Wormhole's 12 DRAM views "
      "and does not support other architectures.", required_cores, num_cores);
  ```

**Proposed fix.** Either (a) add a Blackhole program factory using the BH DRAM-bank layout, or (b) generalize the ring algorithm so the kernel works on any `num_dram_banks` rather than the hardcoded 12.

**Repro.** TBD.

---

## 6. [feature] `experimental.deepseek_moe_reduce_scatter` — add Blackhole program factory

**Pre-filing check:** novel. No existing issue specifically requests BH support for this kernel. [#42697](https://github.com/tenstorrent/tt-metal/issues/42697) (open: "Check functionality with non 32 UPR") is in-flight but a different axis. **File as new.**

**Context.** Kernel currently registers a Wormhole program factory only; both nightly test variants skip on Blackhole.

**Evidence.**
- `tests/nightly/t3000/ccl/test_deepseek_moe_reduce_scatter.py:145`: `@skip_for_blackhole("Requires wormhole_b0 to run")`.
- `tests/nightly/tg/ccl/test_deepseek_moe_reduce_scatter_6U.py:11`: same decorator.

**Proposed fix.** Add a Blackhole program factory. Our use case is the post-`all_to_all_combine` reduce — input shape `[1, 1, 32, 7168]` BF16 DRAM-interleaved (single tensor, not the vector-of-tensors API the WH op currently takes).

**Repro.** TBD.

---

## 7. [feature] `experimental.moe_compute` (and `moe_gpt`) — support 4×8 mesh + `experts_per_device ≥ 8`

**Pre-filing check:** **already covered.** Tracked under [#43444](https://github.com/tenstorrent/tt-metal/issues/43444) (open, assigned). The `experts_per_device > 3` cap is already removed via [#41797](https://github.com/tenstorrent/tt-metal/issues/41797) (closed PR). **Do not file new** — comment on #43444 with our concrete 4×8 + `experts_per_device=8` use case so the team has a target shape.

**Context.** Test infrastructure is gated to 1x16 or 1x8 torus mesh descriptors via `TT_MESH_GRAPH_DESC_PATH`. Default `experts_per_device=2`, with `experts_per_device_values ∈ {4, 6}` in a few model configs but no tested config at 8. DeepSeek-V3 listed in `MODELS_1x16` is parameterised at `mesh_shape=(1,16)` with `experts_per_device=2` (32 total experts on 16 devices), not our 256 experts on 32 devices.

**Evidence.**
- `tests/nightly/tg/ccl/moe/test_moe_compute_6U.py:33-37` defines `MESH_GRAPH_DESC_{1x16,1x8}` constants.
- Lines 1181-1184 gate the parametrized test suite on `is_mesh_graph_descriptor_set(MESH_GRAPH_DESC_1x16) or is_mesh_graph_descriptor_set(MESH_GRAPH_DESC_1x8)`.
- Line 1818: `@pytest.mark.parametrize("mesh_shape, mesh_device", [((1, 16), (1, 16))], indirect=["mesh_device"])`.
- File:65 default `experts_per_device_values=(2,)`; max value across all listed models is 6 (line 110 `kimi_k25` and line 112 `deepseek_v4_pro`).
- The `prepare_w0_w1_tensor_for_moe_compute` packer's shard maps assume the tested topologies.

**Proposed fix.** Add `mesh_shape=(4,8)` to the parametrized matrix, extend `experts_per_device_values` to include 8, and add a BH program factory for `MatmulMultiCoreReuseMultiCast1DProgramConfig` paths in the moe_compute internal matmuls. Likely needs `prepare_w0_w1_tensor_for_moe_compute` to be generalized to non-1xN meshes.

**Repro.** TBD.

---

## 8. [feature] `experimental.ccl.moe.selective_reduce_combine` — support 4×8 mesh

**Pre-filing check:** **already covered** under the same [#43444](https://github.com/tenstorrent/tt-metal/issues/43444) umbrella (the kernel is part of the fused MoE BH path). **Do not file new** — fold into the same #43444 comment as draft #7.

**Context.** Same `TT_MESH_GRAPH_DESC_PATH=1x8/1x16` gating as #7. The op would be the natural replacement for our `all_to_all_combine + reduce_scatter` pair (combined ≈ 2,533 μs in the E0 baseline; ≈ 1,228 μs after the surrounding fusions landed).

**Evidence.**
- `tests/nightly/tg/ccl/moe/test_selective_combine_6U.py:25,672,681` — uses `is_mesh_graph_descriptor_set` to gate on 1x8 / 1x16 descriptors.

**Proposed fix.** Add `mesh_shape=(4,8)` to the parametrized matrix and a BH program factory. Likely shares enough internal CCL plumbing with `moe_compute` (#7) that a single BH cluster-axis-aware refactor unblocks both.

**Repro.** TBD.
