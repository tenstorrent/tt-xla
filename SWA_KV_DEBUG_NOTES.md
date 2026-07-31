# Sliding-window KV accounting — debug notes

Supplementary detail behind the two sliding-window KV issues filed as follow-ups to
#5727 / #5786 / #5834. The issues carry the symptoms and repros; this file carries
the derivations, the dead ends, and the reasoning behind the proof-of-concept
commits on this branch. **The PoC commits are exploratory** — they are here as
evidence that the diagnoses are actionable, not as a proposed fix.

Branches:

| branch | tip | purpose |
|---|---|---|
| `kmabee/issue_5727_followup` | `b2c90b4c3` | reproduce on this. The two sliding-window commits are back-to-back on an otherwise-untouched base, so `git checkout 591d66f52 -- integrations/vllm_plugin/` isolates *only* the sliding-window change |
| `kmabee/issue_5727_followup_debug` | `+ 2 commits` | the same, plus the PoC patches |

**This branch pins vLLM 0.25.1 and is not current main.** On main six PRs sit
between the two sliding-window commits — including a vLLM 0.26 uplift (#5827),
serving telemetry (#5808) and DP+TP fixes (#5801) — so rolling back "just the
sliding-window change" there is not possible, and main's plugin will not run
against this venv's vLLM. All measurements below were taken at `b2c90b4c3`;
`worker.py` and `swa_cache_utils.py` are byte-identical to main's, so the KV-sizing
path is the same one main has.

Hardware for everything: `google/gemma-4-31B-it` (50 sliding + 10 full-attention
layers, `sliding_window=1024`), 4x p300c, 31.88 GiB/chip, via
`tests/benchmark/test_vllm_benchmarks.py::test_vllm_tp_benchmark[gemma4-31b-it-tp]`.
`TTXLA_LOGGER_LEVEL=INFO` is required to see the reserve/budget lines.

---

## 1. Read this first: two theories that are wrong

Both were plausible and both cost time. They are recorded so the next person does
not re-derive them.

### 1.1 "`group_size` becomes 50 and undersizes the pool" — no

The first hypothesis (and the one in the #5727 comment thread) was that
`get_kv_cache_config_from_groups`'s general case computes
`group_size = max(len(g.layer_names) for g in groups)` = 50 (the sliding group), and
so divides the budget by a number unrelated to the 10 real full-attention layers.

`_get_kv_cache_groups_uniform_page_size` splits by `min_num_layers`, so 50 sliding +
10 full becomes **6 groups of 10**, not one 50 and one 10. `group_size` is therefore
**10**, which happens to equal the real full-attention layer count. Measured pool
utilization is **19.22 of 19.23 GiB (100%)**. The divisor strands nothing here.

(It is a coincidence of n:1 sliding-heavy models — see §5.3.)

### 1.2 "The hybrid path pads the full layers' page, doubling their cost" — no

Second hypothesis: the two layer types have different geometry (sliding 16 heads x
256 = 524,288 B/page; full 4 x 512 = 262,144), vLLM requires a uniform page size
across groups, so the full layers must be padded 2x.

`unify_kv_cache_spec_page_size` prefers **raising `block_size`** when the pages
divide evenly, so the full layers get `block_size=64` and page 524,288. Their
per-request cost is `2 x 524,288` = 1 MiB — *identical* to `4 x 262,144`. No memory
is wasted by unification. Verify with:

```python
groups = K.get_kv_cache_groups(cfg, spec)   # -> FullAttentionSpec block=64, page=524288
```

---

## 2. Problem 1 — the ring costs more than full attention below the window

### 2.1 Mechanism

A ring costs `align8(cdiv(min(window, max_model_len), block) + 1)` blocks per user
per layer; full attention costs `cdiv(max_model_len, block)`. The `+1` covers a
window straddling a block boundary and the round-up to 8 satisfies the 32-byte
page-table stick alignment. When `max_model_len <= sliding_window` the window never
clips anything, so both hold the same tokens and the ring's overhead is pure loss:

| max_model_len | full | ring | |
|---|---|---|---|
| 128 | 4 | **8** | 2.00x worse |
| 512 | 16 | 24 | 1.50x worse |
| 1024 | 32 | 40 | 1.25x worse |
| 2048 | 64 | 40 | ring wins |
| 131072 | 4096 | 40 | ring wins 102x |

Crossover is exactly `max_model_len > sliding_window`. **Reach is model-dependent**:
gemma-4's window is 1024, but `Ministral-8B-Instruct-2410` has `sliding_window=32768`,
so for that model the condition holds at *every* `max_model_len` up to 32K.

### 2.2 Exact reconciliation of the numbers in #5727

```
budget = 31.88 GiB/chip x 0.20 gmu x 4 (kv_shard_factor) = 27,380,416,512 B

before #5786 (specs unified -> one UniformTypeKVCacheSpecs group of 60 layers,
              page = sum of the real per-layer pages):
  50*524288 + 10*262144 = 28,835,840 B per block-tuple
  27,380,416,512 // 28,835,840 = 949 blocks
  949 * 32 = 30,368 tokens ; 949 / cdiv(128,32) = 237.25x          [matches]

after #5786 (real hybrid groups, rings reserved off the top):
  rings: 50 * (8 * 32 + 1) * 524288 = 6,736,101,376 B (6.27 GiB, 24.6% of budget)
  pool:  (27,380,416,512 - 6,736,101,376) // (524288 * 10) = 3,937 blocks
  per-request denominator = 5 sliding groups x 5 blk + 1 full group x 2 blk = 27
  3,937 / 27 = 145.8x ; 145.81 * 128 = 18,664 tokens               [matches]
```

Note the second cost: the same `+1` straddle block is charged again in the
concurrency denominator, once per sliding group — 27 blocks/request (270
layer-blocks) against the flat path's 4 (240 layer-blocks), +12.5%.

### 2.3 A second, independent bug found in the same place

With the hybrid manager disabled, vLLM runs `unify_hybrid_kv_cache_specs()` and the
model runner allocates **no** rings — but the worker still reserves ring bytes,
because it sizes the reservation from its own *pre*-unification spec copy. Those
bytes are stranded. Isolated cleanly by two runs of the PoC: with only the
profitability gate, `178.75x`; with the reservation skip as well, `237.25x`.

**The non-obvious trap here:** you cannot simply skip the reservation whenever the
flag is set. `unify_hybrid_kv_cache_specs` **early-returns on a uniform spec dict**,
so for an all-sliding model (every layer sliding — Mistral-style, not gemma-4's
interleave) the specs are *not* rewritten, the runner *does* allocate rings, and
skipping the reservation would under-reserve. The PoC therefore reserves against
`unify_hybrid_kv_cache_specs()` run over a **copy**, so worker and runner agree by
construction. Verified over all four combinations (mixed/uniform x flag on/off).

---

## 3. Problem 2 — the 64K regression

### 3.1 Identifying the historical configuration

The regressed config was remembered as `gpu_memory_utilization=0.9` with "~1.04x
concurrency", from before #5731 scaled the budget by `kv_shard_factor`. Two facts
pin it down:

* #5731 means the same device memory is `gmu = 0.9 / 4 = 0.225` today; both give a
  **28.69 GiB** accounting budget.
* **The "1.04x" is a fingerprint of a 1-byte KV cache.** With bf16 the flat path
  needs 55.0 GiB and could never have fit in 28.69; with `bfp_bf8` it lands at
  exactly 1.04x / 68,352 tokens. Confirmed on hardware.

### 3.2 Mechanism

`SlidingWindowSpec.max_admission_blocks_per_request` bounds one request at
`sliding_window - 1 + max_num_batched_tokens` tokens — i.e. it reads
`max_num_batched_tokens` as the largest chunk **one** request can be scheduled.
Under TT chunked prefill that value is the *batch-wide* budget
(`tt_prefill_chunk_size * max_num_seqs` = 1024 x 32 = 32,768), while
`AscendScheduler` caps each individual request at `tt_prefill_chunk_size` = 1024
("so one long prompt can't eat the whole budget and serialize others"). Each
sliding group is therefore charged `cdiv(1023 + 32768, 32) + 1 = 1057` blocks where
`cdiv(1023 + 1024, 32) + 1 = 65` suffices — a **16x** over-charge on a bound the
scheduler already guarantees cannot be reached.

That bound feeds `_max_memory_usage_bytes_from_groups`, the startup admission check.

### 3.3 The two constraints were contradictory, which is why no GMU worked

| | admission floor | DRAM ceiling (weights 14.57 GiB/chip + 64K activations) | window |
|---|---|---|---|
| before | gmu >= **0.487** | gmu <= ~0.40 | **empty** |
| after the PoC | gmu >= **0.297** | gmu <= ~0.40 | non-empty |

Raising GMU could not help (the reserve is GMU-independent and the requirement was
fixed); lowering it made things strictly worse. Measured, `59cd8a0b0`-equivalent:

| gmu | pool after reserve | requirement |
|---|---|---|
| 0.44 | 24.83 GiB | 30.81 GiB |
| 0.40 | 19.73 GiB | 30.81 GiB |
| 0.30 | 6.98 GiB | 30.81 GiB |
| 0.25 | **0.60 GiB** | 30.81 GiB |

Two independent ways of shrinking *only the requirement* made the same shape run,
which is what localised the bug: `prefill_chunk_size=256` (mnbt 32,768 -> 8,192)
passed at 1.64x / 107,209 tokens, and a 1-byte KV cache halved both sides.

### 3.4 Why the PoC patches the spec method and not the caller

`get_manager_for_kv_cache_spec` (`single_type_kv_cache_manager.py:~1489`) calls the
*same* method to build the **runtime** admission cap, and vLLM's own comment warns:

> "cap the per-request reservation here so admission matches the startup pool sizer
> ... Drift between the two would re-introduce the deadlock from issue #39734 or,
> worse, mid-prefill OOM."

An earlier PoC attempt patched only `max_memory_usage_bytes` (the startup sizer),
which would have created exactly that drift — pool sized for 65 blocks/request while
the runtime let each request reserve up to 1057. Patching the spec method keeps both
consistent by construction.

One further wrinkle: `check_and_update_config` runs in the **front-end** process
(its log line has no `(EngineCore pid=...)` prefix) while both consumers of the bound
run in **EngineCore**. The per-request chunk therefore has to be published in
`TTWorker.__init__` as well, which runs in EngineCore before either consumer.

---

## 4. Measurements

Variants: **A** = `591d66f52` (before sliding window), **B** = `b2c90b4c3` (as
merged), **C** = B + both PoC commits. Only `integrations/vllm_plugin/` is swapped.

### Short context (`max_model_len=128`, batch 32)

| gmu / KV dtype | A | B | C |
|---|---|---|---|
| 0.40, bf16 | 60,768 tok / 474.75x | 43,420 / 339.22x | 60,768 / 474.75x |
| 0.44, bf16 | 66,816 / 522.00x | 48,374 / 377.93x | 66,816 / 522.00x |
| 0.30, `bfp_bf8` | 91,136 / 712.00x | 68,181 / 532.67x | 91,136 / 712.00x |

C lands exactly on A, and that is the **ceiling**, not a partial recovery: an ideal
zero-overhead SWA bound is 237.38x against the flat path's 237.25x at gmu 0.2 — a
0.05% gap that is only `num_blocks` flooring. Below the window nothing can beat
plain full attention.

### Long context (`max_model_len=65536`)

| config | A | B | C |
|---|---|---|---|
| batch 32, `bfp_bf8`, gmu 0.225 | 68,352 / **1.04x** | **fails to start** | 259,666 / **3.96x** |
| batch 32, `bfp_bf8`, gmu 0.30 | 91,136 / 1.39x | 96,210 / 1.47x | **449,958 / 6.87x** |
| batch 32, bf16, gmu 0.40, chunk 1024 | fails (needs 55.0 GiB) | fails (needs 30.81, has 19.73) | admits at 2.99x, then activation DRAM OOM |
| batch 32, bf16, gmu 0.40, chunk **256** | — | 1.64x / 107,209 | unchanged |
| **batch 8**, bf16, gmu 0.44 | 1.02x / 66,816 | **4.00x / 262,356** | identical to B |

The batch-8 row is #5786's own win (3.9x) and C is byte-identical to B there — at
64K the ring genuinely is cheaper, so the profitability gate leaves it alone.

### Highest reached

**663,715 KV tokens / 10.13x at 64K**, batch 10, `bfp_bf8`, gmu 0.30 — see §5.2.

---

## 5. Open, not addressed by the PoC

### 5.1 Unchecked page-table floor

The runner builds the full-attention page table `cdiv(max_model_len, block_size)`
columns wide, so the shared pool must hold at least that many blocks or
`ttnn.paged_update_cache` fails validation:

```
page_table_val.padded_shape()[1] <= cache_tensor.padded_shape()[0]
```

Nothing checks it at startup — it surfaces ~15 minutes into a run. It is *not*
implied by the admission check, which counts the full group at the unified
`block_size=64` (1,024 blocks) and so passes at pool sizes the page table cannot
address. Hit at bf16 / 64K / gmu 0.30 and 0.32 (pool 1,429 and 1,952 blocks vs a
2,048-wide table). Cheap hardening: assert it during KV sizing.

### 5.2 Ring reserve is sized for `max_num_reqs`, not residency

```
reserve = n_sliding_layers * (window_blocks * max_num_reqs + 1) * page_size_bytes
```

One window-sized sub-ring per batch slot. At 64K the achievable concurrency is set
by the full-attention layers on the shared pool — about 7 at gmu 0.30, not 32 — so
~25 of the 32 sub-rings can never be occupied, and the memory is taken off the top
of the budget. Consequence, measured, both runs passing:

| `max_num_seqs` | reserve | pool | concurrency | KV tokens |
|---|---|---|---|---|
| 32 | 15.64 GiB | 22.62 GiB | 6.87x | 449,958 |
| **10** | **4.90 GiB** | **33.35 GiB** | **10.13x** | **663,715** |

**Lowering the batch raises usable concurrency 1.47x** — the opposite of the
intuitive direction. Right-sizing projects to 6.87x -> ~10.4x at gmu 0.30 and
1.06x -> ~5.2x for bf16.

It does **not** free DRAM for activations: rings and pool both come from the same
`gmu x DRAM x kv_shard_factor` budget and vLLM grows `num_blocks` to fill whatever
the reserve leaves. Its help for the DRAM-bound bf16 chunk-1024 case is indirect —
with right-sized rings, bf16 64K at 2x needs only **gmu 0.134**, leaving 13.0
GiB/chip for activations instead of the ~6 that OOM'd.

A fix cannot just shrink the rings: `assign_ring_slots` raises once slots are
exhausted (`test_assign_ring_slots_exhausted_pool_raises`). It needs admission gated
on free ring slots, or `max_num_seqs` derived from what the pool can serve — plus,
meanwhile, a warning when the reserve provisions far more slots than the pool can
fill.

### 5.3 Latent: shared-pool over-allocation

TT allocates `n_full * num_blocks * page` while vLLM sizes `num_blocks` by
`group_size`; they agree only while `group_size == n_full`. True for every n:1
sliding-heavy model we run, but a 1-sliding:11-full model would over-allocate 11x.
No repro today; worth a loud startup check rather than a mystery DRAM OOM.

### 5.4 Adjacent, tracked separately

The MLA KV budget over-commit (`kv_cache_shard_factor()` inflating the budget for a
*replicated* MLA latent cache, from #5731) touches the same reserve loop for a
different reason — `SlidingWindowMLASpec` subclasses `SlidingWindowSpec`, so SWA-MLA
layers enter it and are reserved in un-sharded units. **The PoC inherits that hole**:
it neither fixes nor worsens it. Coordinate if both are fixed — that issue is about
byte *units*, §5.2 is about slot *count*.

---

## 6. Context needed to read any GMU number here

* **GMU applies to TOTAL device DRAM, not free DRAM.** The worker's `profiled` is
  hardcoded 0 (the `#1414` TODO), so `gpu_memory_utilization` does not account for
  weights. gemma-4-31B's weights are **14.57 GiB/chip of 31.88 (45.7%)**, which caps
  usable GMU near **0.44**. `gmu 0.5` cannot work for this model.
* **#5731 equivalence**: a historical `gmu 0.9` under one-chip sizing is `gmu 0.225`
  today.
* **Reported "maximum concurrency" is a relative signal, not serving capacity.** The
  `max_model_len=128` figures (474x, 712x) far exceed `max_num_seqs=32`. They are
  valid for comparing KV-pool sizing — which is what regressed — not achievable
  concurrency.

---

## 7. Reproducing

```bash
cd <tt-xla> && source venv/activate
git checkout kmabee/issue_5727_followup          # or _debug for the PoC

# roll back to before the sliding-window change (plugin only, no rebuild)
git checkout 591d66f52 -- integrations/vllm_plugin/
# ... run ...
git checkout HEAD -- integrations/vllm_plugin/    # restore
```

```bash
# Problem 1: short context.  A: 60,768/474.75x   B: 43,420/339.22x
TTXLA_LOGGER_LEVEL=INFO TT_BENCHMARK_GMU=0.40 \
pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_vllm_tp_benchmark[gemma4-31b-it-tp]"

# Problem 2: 64K.  A: passes at 1.04x   B: fails in ~50s
TTXLA_LOGGER_LEVEL=INFO TT_BENCHMARK_MAX_MODEL_LEN=65536 TT_BENCHMARK_BATCH_SIZE=32 \
TT_BENCHMARK_GMU=0.225 TT_BENCHMARK_KV_CACHE_DTYPE=bfp_bf8 TT_BENCHMARK_PREFILL_CHUNK_SIZE=1024 \
pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_vllm_tp_benchmark[gemma4-31b-it-tp]"

# §5.2: lowering the batch raises concurrency (10.13x vs 6.87x at batch 32)
TTXLA_LOGGER_LEVEL=INFO TT_BENCHMARK_MAX_MODEL_LEN=65536 TT_BENCHMARK_BATCH_SIZE=10 \
TT_BENCHMARK_GMU=0.30 TT_BENCHMARK_KV_CACHE_DTYPE=bfp_bf8 TT_BENCHMARK_PREFILL_CHUNK_SIZE=1024 \
pytest -svv "tests/benchmark/test_vllm_benchmarks.py::test_vllm_tp_benchmark[gemma4-31b-it-tp]"
```

Runtime: ~5 min per `mml=128` cell, ~13-17 min per passing 64K cell, ~1 min per cell
that fails the KV admission check.

Unit tests for the PoC (seconds, no hardware):

```bash
venv/bin/python -m pytest -q tests/integrations/vllm_plugin/ -m cpu   # 50 passed
```

Full per-configuration sweeps, ~35 run logs and offline calculators that reproduce
the arithmetic above without hardware were kept locally and can be shared on request.
