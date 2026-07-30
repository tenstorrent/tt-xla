# DeepSeek Sparse Attention (DSA) — Blackhole bring-up handoff

**Audience:** a Claude Code instance running on a machine with Blackhole silicon.
**Author:** a Claude Code instance on a Wormhole machine (n300 llmbox, 8 chips).

Everything below was brought up and validated on **Wormhole**, where the three DSA
TTNN kernels **do not exist**. Wormhole therefore only ever ran the composites'
inlined *primitive decompositions*. Your job is the part this machine structurally
could not do: run DSA on the real kernels and confirm the decompositions disappear.

Companion doc: [`dsa-tt-mlir-changes.md`](./dsa-tt-mlir-changes.md) — the tt-mlir
submodule changes, plus gaps G1–G6 with a concrete mask-vs-gather analysis. Read its
"Related tt-mlir / tt-metal gaps" section before touching decode performance.

---

## 1. TL;DR — state and your objective

| | status |
|---|---|
| Op wrappers (`tt.indexer_score_dsa`, `tt.topk_large_indices`, `tt.sparse_sdpa`) | done, committed |
| `TTIndexer` (TT replacement for vLLM's CUDA-only `Indexer`) | done, committed |
| Sparse prefill + sparse decode in `TTMLAAttentionBackendImpl` | done, committed |
| Indexer KV-cache spec + binding | done, committed |
| E2E, 3 layers, dense param | **PASSES** on Wormhole |
| E2E, 3 layers, sparse param (`index_topk=128`) | **PASSES** on Wormhole (decomposition) |
| E2E, 3 layers, production `index_topk=2048` | **SKIPPED** — OOMs on Wormhole's decomposition |
| A/B numerics test (sparse must equal dense) | **SKIPPED** — reported stall, see §9.2 |
| Kernel path (`ttnn.*` promoted ops) | **NEVER EXERCISED** — needs you |

**Primary objective:** run `test_tensor_parallel_generation_deepseek_v32_3l` on
Blackhole and prove the three composites promote to TTNN kernels instead of inlining
decompositions (§5, §6). Then work §9's open items.

---

## 2. Repository state

### tt-xla

```
branch  hshah/dsa-vllm-latest
HEAD    efba06ee2  "add support for DSA"   (21 files, +4682/-380)
```

All DSA work is **committed**. Working tree was clean except `docs/` at handoff time.

Files that matter:

| path | what |
|---|---|
| `python_package/tt_torch/custom_ops.py` | the three `torch.library.custom_op` wrappers + CPU references |
| `integrations/vllm_plugin/vllm_tt/layers/dsa_indexer.py` | `TTIndexer`, the sparse predicates, `install_tt_indexer` |
| `integrations/vllm_plugin/vllm_tt/attention_impls/attention_mla.py` | `_forward_prefill` sparse branch, `_forward_decode_sparse` |
| `integrations/vllm_plugin/vllm_tt/model_runner.py` | indexer KV-cache spec + the `bind_kv_cache` split |
| `integrations/vllm_plugin/vllm_tt/platform.py` | `use_sparse` gate opened, `dsa_mode` config |
| `integrations/vllm_plugin/vllm_tt/vllm_distributed_utils.py` | `disable_tp` skip (keeps the indexer replicated) |
| `tests/torch/models/deepseek_v3_2_exp/build_weight_cache.py` | `--vllm-keys` bf16 checkpoint builder |

### tt-mlir submodule — ⚠️ UNCOMMITTED CHANGES

```
path    third_party/tt-mlir/src/tt-mlir
branch  hshah/all-dsa-ops
commit  7d4fe61b98
state   4 files MODIFIED, NOT COMMITTED  (110 insertions, 34 deletions)
```

The three DSA ops themselves live in commit `7d4fe61b98` on `hshah/all-dsa-ops`.
On top of that there is an **uncommitted** fix you must reproduce:

```
lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp   | 101 +++++++++----
test/ttmlir/Conversion/StableHLOToTTIR/transformer/sparse_sdpa.mlir       | 17 +-
test/ttmlir/Dialect/TTNN/transformer/sparse_sdpa_decomposition_bsz.mlir   | 13 +-
test/ttmlir/Dialect/TTNN/transformer/sparse_sdpa_decomposition_wh.mlir    | 13 +-
```

**What it does:** rebuilds the `sparse_sdpa` *decomposition*'s sparsity mask with a
`scatter` instead of an `[1, S, TOPK, T]` slot-hit broadcast. Without it the
decomposition allocates `O(S·TOPK·T)` — at `S=TOPK=T=1024` that is 1.07e9 elements
(~4.3 GB) in one intermediate and the run never finishes.

**Relevance to you:** strictly a *decomposition* fix, so it does not affect the
Blackhole kernel path. Reproduce it anyway — it keeps Wormhole comparisons possible
and the lit tests green. A verbatim, ready-to-apply patch is in **Appendix A** of
`dsa-tt-mlir-changes.md`; I verified it matches the live diff exactly (modulo two
trailing-space-only lines, so use `git apply --whitespace=fix` or `patch -l`).

```bash
cd third_party/tt-mlir/src/tt-mlir
git log --oneline -1                       # expect 7d4fe61b98 on hshah/all-dsa-ops
git apply --whitespace=fix /path/to/appendix-a.patch
git diff --stat                            # expect the 4 files above
```

If your submodule is at a different commit, **stop and reconcile first** — the three
DSA ops may not exist there at all.

---

## 3. Environment and build

```bash
cd <tt-xla>
source venv/activate            # sets TTXLA_ENV_ACTIVATED, TTMLIR_TOOLCHAIN_DIR
cmake -G Ninja -B build
cmake --build build             # rebuild after applying the tt-mlir patch
```

The vLLM plugin is a separate editable package under `integrations/vllm_plugin`
(`vllm_tt`), pinned to `vllm==0.22.1`. Useful during debugging:

```bash
export TTXLA_LOGGER_LEVEL=DEBUG
```

---

## 4. Prerequisite: the bf16 weight cache

The published DeepSeek-V3.2 checkpoint is **fp8 block-quantized**, which TT cannot
run — vLLM routes it to `Fp8LinearMethod` (CUDA-only block GEMMs) and tt-xla does not
dequantize fp8 on load. You need a bf16 copy of the first 3 layers with HF parameter
names preserved:

```bash
python tests/torch/models/deepseek_v3_2_exp/build_weight_cache.py \
    --repo deepseek-ai/DeepSeek-V3.2-Exp --n-layers 3 --vllm-keys
```

- Only the shards holding those layers download (~10 GB, not the full ~690 GB).
- Output lands at
  `~/.cache/huggingface/tt_xla_dequant_cache/deepseek-ai--DeepSeek-V3.2-Exp_3layers_vllm`
  and is **6.8 GB** on disk.
- `--vllm-keys` is what preserves HF names, drops `weight_scale_inv`, and strips
  `quantization_config`. Without it you get the torch-test layout, which vLLM cannot
  load.
- If it is missing the test **skips** with the exact command in the skip message, so
  a skip is not a failure.

Config facts (stock, from the model's `config.json`): `num_attention_heads=128`,
`kv_lora_rank=512`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`,
`q_lora_rank=1536`, `hidden_size=7168`, `index_head_dim=128`, `index_n_heads=64`,
**`index_topk=2048`**, `first_k_dense_replace=3`, `num_hidden_layers=61`.

`first_k_dense_replace=3` means layers 0–2 are the **dense-MLP** layers, so a
3-layer run covers MLA + indexer + rope + weight loading and **no MoE**. Use 4 layers
if you ever want the first MoE block.

---

## 5. Running the E2E test

```bash
pytest -svv "tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py::test_tensor_parallel_generation_deepseek_v32_3l"
```

Three params: `[dense]`, `[sparse-topk128]`, `[sparse-topk2048]` (currently skipped).
Select one with e.g. `...::test_tensor_parallel_generation_deepseek_v32_3l[sparse-topk128]`.

| param | `index_topk` | `model_len` | prompt | sparse? |
|---|---|---|---|---|
| `dense` | stock 2048 | 128 | 10 tok | no — `128 < 2048` |
| `sparse-topk128` | overridden to 128 | 256 | 234 tok | **yes**, prefill *and* decode |
| `sparse-topk2048` | stock 2048 | 4096 | 2106 tok | **yes** — skipped on WH, see §9.1 |

The test asserts only (a) tokens were produced and (b) **which DSA custom calls
reached the exported StableHLO**. Output text is expected gibberish — 3 of 61 layers
feed `lm_head` nothing resembling the real model — so coherence is deliberately not
asserted. The op-emission assertion is the real content: the sparse/dense split is a
static Python branch, so without it a silent fall back to dense would still pass.

### ⚠️ 5.1 The mesh is hardcoded to `[2, 4]` — you will likely need to change it

`additional_config["mesh_shape"] = [2, 4]` and the marker is `@pytest.mark.llmbox`,
i.e. **8 chips**. On a Blackhole quietbox (`bhqb`, 4 × p300c) that will not run.

Use `[1, 4]` and swap the marker. This is not arbitrary — `tt.sparse_sdpa` requires
`heads >= 32 && heads % 32 == 0` **per device, post-Shardy**:

| mesh | model axis | heads/device | legal? |
|---|---|---|---|
| `[2, 4]` (llmbox) | 4 | 32 | ✅ exactly at the limit |
| `[1, 4]` (bhqb) | 4 | 32 | ✅ |
| `[8, 4]` (bh_galaxy) | 4 | 32 | ✅ |
| anything with model axis 8 | 8 | 16 | ❌ **silently decomposes** |

Relevant markers (`pytest.ini`): `bhqb` = blackhole quietbox (4-chip p300c),
`bh_galaxy` = blackhole galaxy, `single_device` covers `wormhole_b0` and `p150`.
The already-present A/B test (§9.2) parametrizes `[1, 4]`/`bhqb` and `[8, 4]`/`bh_galaxy`
— copy those.

### 5.2 Expected runtime and output (Wormhole reference)

Both non-skipped params passed: **`2 passed in 97.76s`**. Per param: weights load in
~0.6 s (54 tensors), 8 warmup graphs compile, then generation. `dense` alone was
`1 passed in 79.85s`. Blackhole should be comparable or faster.

```
dense           token_ids: [30587, 122095, 122095, ...]  'ODO/questions/questions...'
sparse-topk128  token_ids: [119103, 13484, 122095, ...]  'hurican/questions/questions...'
```

The two differ, correctly: at `index_topk=128 < 256` sparse genuinely drops keys.
Do **not** treat divergence as a bug here — that is what the A/B test (§9.2) is for.

Benign log lines, not failures:
- `safe_mark_sharding: dim 0 (size 1) not divisible by mesh axis 'batch' (size 2); replicating instead` — expected at `max_num_seqs=1`.
- On Wormhole only: `[TT] DSA kernels are Blackhole-only; ... composites inline their primitive decompositions`. **On Blackhole this warning must NOT appear.** Its absence is your first signal that promotion is live.

---

## 6. What DSA TTNN ops to look for — the core verification

### 6.1 The pipeline

```
tt.<op> (stablehlo.custom_call)
  → ttcore.composite {composite_name, decomposition = @<op>_decomp}
  → TTNNResolveComposites:
      promotionGuard passes (requireBlackhole) AND validate passes
        → ttnn.<op>                    ← WHAT YOU WANT
      otherwise
        → inline @<op>_decomp          ← what Wormhole always gets
```

**Failure is silent.** A failed guard or a failed TTNN-verifier validation yields
`nullptr` and falls back to inlining. There is no error, no warning — just slower,
and for `topk_large_indices` semantically different (the decomposition's `ttir.topk`
does not reproduce the `-inf → 0xFFFFFFFF` sentinel contract; tt-xla compensates with
an explicit index repair, which the kernel path makes unnecessary).

### 6.2 Verification recipe

The engine core runs in a **child process**, so in-process counters see nothing.
Grep the exported MLIR. `additional_config["export_path"]` is already wired into the
test (`tmp_path/ir`); pytest keeps the last three tmp dirs under
`/tmp/pytest-of-$USER/pytest-<n>/`.

```bash
IR=/tmp/pytest-of-$USER/pytest-<n>/test_tensor_parallel_generatio<k>/ir/irs

# MUST be > 0 on Blackhole
for op in ttnn.indexer_score_dsa ttnn.topk_large_indices ttnn.sparse_sdpa; do
  printf "%-28s %s\n" "$op" "$(cat $IR/ttnn*.mlir | grep -c "$op")"
done

# MUST drop to 0 in the DSA graphs on Blackhole
for op in ttnn.relu ttnn.scatter ttnn.softmax; do
  printf "%-28s %s\n" "$op" "$(cat $IR/ttnn*.mlir | grep -c "\"$op\"")"
done

# MUST be 0 on both (an unresolved composite is a pass bug)
grep -c ttcore.composite $IR/ttnn*.mlir
```

### 6.3 Wormhole baseline to diff against (`sparse-topk128`, measured)

24 `ttnn*.mlir` dumps = 12 graphs × 2 stages. Exactly **two** graphs carry DSA — one
sparse prefill, one sparse decode — each with **3 sites (one per layer)**:

| op | Wormhole (decomposition) | Blackhole (expected) |
|---|---|---|
| `ttnn.indexer_score_dsa` | **0** | **3 per DSA graph** |
| `ttnn.topk_large_indices` | **0** | **3 per DSA graph** |
| `ttnn.sparse_sdpa` | **0** | **3 per DSA graph** |
| `ttnn.relu` | 12 (3 per DSA graph × 4 dumps) | **0** |
| `ttnn.scatter` | 12 | **0** |
| `ttnn.softmax` | 12 | **0** |
| `ttnn.ge` | 28 | reduced |
| `ttnn.arange` | 18 | reduced |
| `ttnn.topk` | 46 | **~34** — see caveat |
| `ttcore.composite` | 0 | 0 |

**Caveat on `ttnn.topk`:** it is *not* a clean signal. Of the 46, only 12 come from
the `topk_large_indices` decomposition; the other ~34 are the **sampler's** topk and
will remain. Either scope the grep to the two DSA graphs, or just use presence of
`ttnn.topk_large_indices` as the positive signal.

`ttnn.relu`, `ttnn.scatter` and `ttnn.softmax` appeared **only** in the two DSA graphs
on Wormhole, so those three are clean signals — if any survives in a DSA graph on
Blackhole, that composite did not promote.

### 6.4 If a composite does not promote

Check in this order:

1. **Arch.** All three kernels hard-`TT_FATAL` on Blackhole in tt-metal:
   `sparse_sdpa_device_operation.cpp:73`, `topk_large_indices_device_operation.cpp:31`,
   `indexer_score_device_operation.cpp:235` (reason at :232 — "relies on BH
   fast-untilize + custom BH LLK paths"). Confirm `ttcore.SystemDescAttr` really says
   blackhole.
2. **Per-device head count** — §5.1. The op wrapper only sees the *global* count and
   cannot check this.
3. **A TTNN verifier constraint** — §7. Validation failure ⇒ silent fallback.
4. **Offline reproduction** without hardware:
   ```
   ttmlir-opt --ttir-to-ttnn-backend-pipeline="mock-system-desc-arch=blackhole composite-resolution=force-promote"
   ```
   `force-promote` turns the silent veto into a hard error, which is the fastest way
   to see *which* constraint bit.

---

## 7. Constraint checklist (violations ⇒ silent decomposition)

| op | constraint | V3.2 @ mesh model-axis 4 |
|---|---|---|
| `indexer_score_dsa` | rank 4 all operands; **B == 1**; `key == [B,1,T,D]`; `weights == [B,Hi,Sq,1]`; `result == [B,1,Sq,T]`; query/key same elem type | B==1 via per-user loop ✅ |
| `topk_large_indices` | input **bf16**; result **ui32**; `k ∈ [16, 2048]`; **`k % 16 == 0`**; `input[-1] >= k` | k=2048 at the ceiling ✅ |
| `sparse_sdpa` | **B == 1**; **`H >= 32 && H % 32 == 0`**; `K_DIM % 32 == 0`; `0 < v_dim <= K_DIM`, `v_dim % 32 == 0`; `k_chunk_size >= 32 && % 32 == 0`; **`TOPK % k_chunk_size == 0`**; indices integral | H=32, K_DIM=576, v_dim=512, chunk=128, 2048%128=0 ✅ |

Both `indexer_score_dsa` and `sparse_sdpa` require **batch 1**, which is why the
indexer and `_forward_decode_sparse` loop per user in Python. That loop unrolls into
the traced graph and scales with `max_num_seqs` — a known cost, see §9.3.

`k_chunk_size` is chosen automatically as the largest of `(128, 64, 32)` dividing
`topk`; for `topk=2048` that is 128, matching the tt-metal default.

**The sentinel contract is load-bearing.** `topk_large_indices` emits `0xFFFFFFFF`
for `-inf` scores; because top-k is sorted descending, sentinels form a *contiguous
tail*. `sparse_sdpa_reader.cpp:94` **binary-searches the first sentinel** to get the
valid-key count and activates only `ceil(nv/k_chunk)` chunks. A non-contiguous tail
would truncate that count and silently drop real keys.

---

## 8. Architecture: how DSA is wired

```
MultiHeadLatentAttentionWrapper.forward            (upstream vLLM)
  ├─ TTIndexer(hidden_states, q_c, positions, rope)      ← replaces CUDA Indexer
  │    ├─ wq_b / wk_weights_proj / k_norm / rope   (bf16, NO fp8 quant)
  │    ├─ tt.paged_fill_cache | tt.paged_update_cache → indexer K cache
  │    ├─ if sparse: tt.indexer_score_dsa → tt.topk_large_indices
  │    └─ publishes indices on an instance slot (NOT vLLM's topk_indices_buffer)
  └─ TTMLAAttention → TTMLAAttentionBackendImpl.forward
       ├─ prefill: tt.sparse_sdpa   (else tt.flash_mla_prefill)
       └─ decode:  gather + tt.sparse_sdpa   (else tt.paged_flash_mla_decode)
```

**Install path:** `TTIndexer` is a subclass of upstream `Indexer` installed by
rebinding `vllm.model_executor.models.deepseek_v2.Indexer`, fired from
`model_runner.load_model` only when `hf_config.index_topk` exists. It rebuilds only
what TT needs (upstream's `__init__` constructs an fp8 cache and `SparseAttnIndexer`),
reusing upstream submodule classes so checkpoint parameter names still resolve.

**Sparse predicates** (`dsa_indexer.py`) — both take Python ints at trace time, so
these are *static* branches, one graph per bucket:
- `dsa_prefill_uses_sparse(seq_len, topk)` → `seq_len >= topk`
- `dsa_decode_uses_sparse(max_seq_len, topk)` → computed from the **page-table
  bucket width**, never from `cache_position` (a runtime value)

Below the threshold, sparse is not merely skipped for speed — it is *impossible*
(`topk_large_indices` needs `input[-1] >= k`) *and* pointless (top-k over ≤ topk
visible keys selects all of them, so sparse ≡ dense causal, exactly).

**`dsa_mode`** (`additional_config`), values `"auto"` | `"dense_decode"` | `"off"`:
- `auto` — sparse wherever the predicates clear
- `dense_decode` — sparse prefill, dense decode (faster today, deviates from trained sparsity)
- `off` — indexer still built and weights still load, but no DSA op emitted

`off` is the A/B and bisect lever. Note the indexer's projections, rope and K-cache
writes run in **all** modes; only scoring/top-k/sparse-attention are gated.

**Indexer KV cache:** bf16, `[num_blocks, 1, block_size, 128]`, via `MLAAttentionSpec`
with `head_size=index_head_dim`. Two extra plumbing facts worth knowing:
`get_kv_cache_spec` needed an explicit `DeepseekV32IndexerCache` branch (it is neither
`Attention` nor `MLAAttention`, so it fell through `else: continue`), and
`bind_kv_cache` had to be wrapped by `bind_kv_cache_allowing_dsa`
(`model_runner.py:174`) because it groups layers by `extract_layer_index` — the MLA
layer and its indexer cache collide on the same index, and upstream `raise`s on
non-CUDA platforms in that case.

---

## 9. Open items, highest value first

### 9.1 Re-enable `sparse-topk2048` under a Blackhole marker ⭐ the main prize

This is the only param that runs DSA at **production `index_topk=2048`**. It is
skipped because Wormhole's decomposition OOMs, and the skip reason says explicitly:
*"Re-enable under a Blackhole marker, where all three composites promote to kernels
and none of these intermediates is materialized."*

What Wormhole hit, for contrast: the inlined `indexer_score_dsa` decomposition
carries a `[1, 64, 4096, 4096]` bf16 intermediate (2 GiB, confirmed in the exported
TTIR), then execution died in `bank_manager` requesting a single **32 GiB** DRAM
buffer on a 12 GiB device. That 32 GiB is 16× the largest tensor in the IR and appears
in no tensor type, so it is runtime scratch inside a TTNN op — `ttnn.topk` over a
4096-wide row with `k=2048` is the prime suspect, and it is precisely what the
Blackhole `topk_large_indices` kernel (which streams in LLK-sized windows) replaces.

**Expect this to just work on Blackhole.** If it does, drop the skip and mark it
`bhqb`/`bh_galaxy`.

⚠️ `model_len` **must be a power of two**: `_adjust_min_token` silently rounds
`min_context_len` up to one while the page table is still sized from `max_model_len`.
2176 was tried and produced a 4096-token bucket against a 68-block page table, dying
with *"Input seq_len (4096) must fit in max_num_blocks_per_seq (68) * block_size (32)"*.
2048 cannot hold the 2106-token prompt, hence 4096.

### 9.2 Re-run the A/B numerics test — the only real correctness gate

`test_tensor_parallel_generation_deepseek_v32_dsa` (same file, currently
`@pytest.mark.skip`). It sets `index_topk` **equal to** the padded prefill length, so
top-k covers every causally visible key and sparse prefill **must** reproduce
`dsa_mode="off"` token-for-token. Everything shipped so far asserts op *emission*;
this asserts op *correctness*. It already parametrizes `[1,4]`/`bhqb` and
`[8,4]`/`bh_galaxy`, and asserts `DSA_TTNN_OPS` are present on the Blackhole param.

Its skip claims a hang: *"All 12 graphs convert to ttnn_runtime MLIR in ~20s, then
execution never completes ... parks in `FDMeshCommandQueue::read_completion_queue`"*,
attributed to "all three DSA ops inside one compiled model graph under TP sharding"
at `DSA_MODEL_LEN` 256 and 1024.

**Treat that attribution as doubtful.** I ran exactly that configuration — same
model, 3 layers, `[2,4]` mesh, 256 prefill, all three DSA ops in one graph under TP —
and it completed in ~50 s. So the stall was either fixed by intervening work
(plausibly the `sparse_sdpa` scatter change, §2) or misattributed. Re-run it before
trusting the skip reason.

### 9.3 Decode scales with context — the real perf work

Sparse decode is **correct but slower than dense decode**, and worsens with context.
Two `O(context)` gathers per layer per step: the MLA latent cache
(`attention_mla.py:_forward_decode_sparse`) and the indexer K cache
(`dsa_indexer.py:_forward_decode`), both because the ops need dense, batch-1 operands.

> ⚠️ **CORRECTION.** The G1 summary below is wrong, and the correction matters because
> it re-scopes the work from a tt-mlir one-liner to a tt-metal feature request.
> Exposing `cache_batch_idx` does **not** remove the gather on a paged cache: the
> parameter assumes a batch-contiguous cache (`kv_batch_page_offset = cache_batch_idx * T`),
> and `sparse_sdpa` requires `kv` ROW_MAJOR while `update_cache` requires the same
> tensor TILE. Details and the actual ask (make `sparse_sdpa` paged-aware, as
> `paged_flash_mla_decode` already is) are in
> [`dsa_blackhole_tt-metal_changes.md`](./dsa_blackhole_tt-metal_changes.md) §2.8, with
> the corrected analysis in [`dsa-tt-mlir-changes.md`](./dsa-tt-mlir-changes.md) §G1.

`dsa-tt-mlir-changes.md` §"G1/G2 addendum" has the full traffic comparison. Summary:
**G1 (expose `cache_batch_idx` on `TTNN_SparseSdpaOp`) is the single change that makes
DSA decode fast rather than merely correct** — it makes traffic proportional to
`top-k` and *flat in context*. tt-metal already accepts the parameter
(`sparse_sdpa.hpp:48-50`); tt-mlir does not expose it and the runtime hardcodes
`std::nullopt`. This is Blackhole-only by nature, so **you are the right machine to
land and measure it.**

### 9.4 Also worth running on Blackhole

```bash
pytest -svv tests/torch/ops/test_dsa_ops.py            # push+single_device (p150 works)
pytest -svv tests/torch/ops/test_dsa_ops_cpu.py        # cpu marker, no hardware
pytest -svv tests/integrations/vllm_plugin/oot_backends/test_dsa_indexer.py
pytest -svv tests/integrations/vllm_plugin/oot_backends/test_dsa_prefill_impl.py
pytest -svv tests/integrations/vllm_plugin/oot_backends/test_mla_attention_impl.py
```

`test_dsa_ops.py` is where production-shape op coverage lives (`index_topk=2048`).
Its device tests are `single_device`, which includes **p150 (Blackhole)** — so they
should exercise real kernels there with no edit. The two `oot_backends` DSA tests use
`@parametrize_arch(["llmbox"])`; they need a Blackhole arch added to run for you.

`tests/utils.py` has `get_torch_device_arch()` → `TTArch.BLACKHOLE` for runtime
arch gating if you need to write kernel-only assertions.

### 9.5 Housekeeping

- `dsa_e2e_as_run_1024.py` at repo root is a scratch driver from the 1024-width
  investigation. Committed, but it is not a test — consider deleting or folding in.
- `vllm_mla_prefill_flow.md` at repo root is an older design note; some line numbers
  are stale.
- G3–G6 in the companion doc are unaddressed. G6 (vLLM's `MLAAttentionSpec.merge`
  not validating `head_size`) is the most damaging and belongs in a vLLM PR.

---

## 10. Traps I hit, so you don't

1. **Silent decomposition fallback.** The single biggest hazard. Always verify via
   exported IR (§6.2); never infer promotion from a passing test.
2. **`torch.uint32` is a prototype dtype.** `topk_large_indices` returns it because it
   is the only dtype that lets the composite promote (the TTNN verifier requires
   `isUnsignedInteger(32)`; si32 converts fine then silently fails validation). But
   CPU `where`/`gather`/comparison/promotion are **unimplemented** — always
   `.to(torch.int64)` before touching an indices tensor.
3. **The engine core is a child process.** In-process op counters see nothing.
4. **`min_context_len` rounds to a power of two** while the page table is sized from
   `max_model_len` — §9.1.
5. **`_exported_ir` globs `{stage}_*.mlir`** under `export_dir/irs`; useful stage
   prefixes are `shlo`, `shlo_compiler`, `ttnn`. Note `shlo*` also matches
   `shlo_compiler*`, so raw counts across stages are inflated — compare *presence*,
   or scope to one stage.
6. **`hf_overrides={"index_topk": N}`** changes no weight shape, so it is a safe way
   to move the sparse threshold. Leave it out to get the stock 2048.
7. **The Wormhole warning is your promotion canary.** If
   *"DSA kernels are Blackhole-only"* appears on a Blackhole run, the arch was
   misdetected — fix that before debugging anything else.
