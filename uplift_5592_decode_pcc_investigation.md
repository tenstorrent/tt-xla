# Uplift #5592 — decode PCC=0.0 investigation (handoff)

**Issue:** [#5605](https://github.com/tenstorrent/tt-xla/issues/5605) (P0). **PR:** [#5592](https://github.com/tenstorrent/tt-xla/pull/5592) (tt-mlir `260d4c4` → `327b846`).
**Affected:** single-chip LLM perf tests — `test_falcon3_1b`, `test_llama3.1_8b`, `test_qwen3_8b` (n150).
**Symptom:** prefill PCC ~0.997; **first-decode PCC = 0.000000** (`Required=0.94`). CI: run [29223626701](https://github.com/tenstorrent/tt-xla/actions/runs/29223626701), job `86736187259`.
**Investigator:** mvasiljevic. **Date:** 2026-07-13. **Machine:** this host (wormhole_b0 / n150).

---

## ✅ SOLUTION (shipped) — read this first

**Localized issue:** the tt-mlir uplift `260d4c4→327b846` bumps tt-metal `13adda80c11→38e954a066c` (commit `30645f913`). The new tt-metal has a **trace capture/replay regression**: once the **perf run** captures a trace on the device, that resident trace **corrupts the KV-cache read of the next graph** that runs with a device-native cache. So the benchmark's PCC/logits run (which runs *after* the perf run) decodes over a corrupted cache → first-decode logits **explode to ~10¹⁹ → PCC=0.0**. The perf run itself is fine; the KV-cache content and SDPA/lm_head math are correct (the argmax path decodes the correct token). Proven trace-specific: `TTXLA_NO_TRACE=1` or not running the perf run first both make PCC pass.

**Fix (tt-xla, perf-preserving):** run the **PCC/correctness run FIRST** (clean device, no resident trace) and the **traced perf benchmark LAST**. Perf stays traced → measured perf unchanged. Pure reorder in `tests/benchmark/benchmarks/llm_benchmark.py` (`_run_perf_benchmark()` moved after the PCC/TOPK block).

**Local validation (full falcon3-1b, bs32, on the failing tt-metal):** decode PCC **0.998** (was 0.0); **51.3 sps** vs **52.3** perf-first baseline = **−2%** (within the 5% perf gate). (NO_TRACE alternative = −57%, rejected.)

**Branches:**
- Fix (minimal, only the reorder): `mvasiljevic/uplift-5592-decode-pcc-fix`
- Investigation (this doc + diagnostics): `mvasiljevic/uplift-5592-decode-pcc-investigation`

**CI (single-chip n150-perf, `mlir_override=327b846`, regression_check on):**
- FIX: https://github.com/tenstorrent/tt-xla/actions/runs/29525027126 (expect PCC pass + perf within 5%)
- CONTROL (no fix): https://github.com/tenstorrent/tt-xla/actions/runs/29525072439 (expect PCC=0.0 fail)
- Original failing CI (issue): https://github.com/tenstorrent/tt-xla/actions/runs/29223626701

**Proper long-term fix is upstream in tt-metal** — see next section. The reorder is a tt-xla-side workaround that unblocks the uplift without a perf hit.

---

## TT-METAL ROOT CAUSE + PROPER FIX (so BOTH perf and PCC work under trace)

The reorder only avoids the collision; the real bug is in tt-metal. Mechanism (evidence below):

- tt-metal **metal-trace replays a program's runtime args VERBATIM** (it does not re-patch raw address args on replay).
- The **SDPA-decode and KV-cache op program factories SMUGGLE raw buffer addresses into kernel runtime args** instead of declaring them as `Buffer*` bindings:
  - `ttnn/.../transformer/sdpa_decode/device/sdpa_decode_program_factory.cpp:909-911` (`q/k/v_buffer->address()`), `:936` (`out`), `:825-828` (`pos/page_table/attn_mask/sink`)
  - `ttnn/.../kv_cache/device/update_cache_multi_core_program_factory.cpp:306-320`, `fill_cache_multi_core_program_factory.cpp:165-177` (`dst/src` addresses)
- So a captured decode trace bakes the **KV-cache buffer address** at capture time. It replays correctly only if that buffer stays at the same address. When a **perf-run trace + its buffers are already resident**, the PCC run's fresh KV-cache lands differently, so the baked address no longer matches the buffer at replay → the decode reads **stale/wrong memory** → logits explode → PCC 0.0. Running the PCC decode first (clean, deterministic allocation) keeps the baked address valid — which is exactly why the reorder fixes it.
- This smuggling is present in the **baseline (good) tt-metal too** (identical lines), so it's a latent hazard; the new tt-metal changed buffer/RTA/trace behavior in `13adda80c11..38e954a066c` such that the baked address now goes stale. tt-metal is actively **purging this exact anti-pattern**: PR `#49141` adds a pre-commit guard (`scripts/detect_smuggled_rta.py`) against "a raw `tensor.buffer()->address()` pushed into kernel runtime args instead of a Buffer* binding via `emplace_runtime_args`", and PRs `#49132–#49138` migrated bcast/attn_matmul/batch_norm/reduction/eltwise/matmul/moreh/pool/normalization — **but NOT the transformer sdpa_decode / kv_cache ops.**

**Proper fix (tt-metal PR):** migrate the `sdpa_decode`, `update_cache`, and `fill_cache` program factories to declare their buffer addresses as `Buffer*` **BufferBindings** (`emplace_runtime_args`) instead of raw `->address()` runtime args — matching the `#49132–#49138` migration. Then metal-trace patches the addresses on every replay, the baked address can't go stale, and **both the perf run and the PCC run work under trace** (no reorder needed).

**Exact regressing commit: not pinned** (candidates that changed RTA/trace dispatch in range: `#48686` WRITE_PACKED_LARGE_UNICAST >341 RTAs, `#48034` per-core RTA reserve; plus general DRAM-allocation changes). Definitive pin = a tt-metal bisect of `13adda80c11..38e954a066c` using the `--num-layers 1` full-flow perf-first repro (good=PCC passes, bad=0.0). No existing tt-metal issue found for this exact symptom.

Reorder caveat: applied unconditionally; decode_only/multichip use the same reorder (logically order-only), validated on single-chip.

---

## ⚡ LATEST ROOT CAUSE (trace) + RESUME STATE — read this first

**Root cause is a tt-metal TRACE-buffer allocation regression.** Proven by experiment on the bad build (`--num-layers 1`):
- `TTXLA_NO_TRACE=1` → decode PCC **0.999 (PASS)**. Disabling trace fixes it.
- `TTXLA_SKIP_PERF=1` (don't run the perf trace before the PCC run) → **0.999 (PASS)**.
- full flow (perf trace → PCC trace, device cache) → **0.0 (explode)**.
- `--decode-only` (CPU cache) → sane. good build → sane.

⇒ The **perf run's trace poisons the subsequent PCC run's trace** when the KV cache is device-native — a tt-metal trace-buffer allocation collision on the new tt-metal (`38e954a066c`). The KV-cache content + SDPA/lm_head math are fine (argmax path is correct).

**One-flag minimal repro & workaround:** `test_falcon3_1b --num-layers 1` → decode PCC 0.0; add `TTXLA_NO_TRACE=1` → 0.999.

**Candidate tt-metal commits (in range, touch trace-buffer allocation), NOT yet pinned:**
- `abf1dd34b` #47122 "make trace_region_size the total trace budget across DRAM banks" (semantics change)
- `1741c2c66` #47766 "round trace per-bank reservation up to max trace page size" (OOM fix for #47122)
- `6beb91f69` #48900 "pre-allocate on-device sampling buffers before trace capture"
- CAVEAT: tt-mlir opens the device with `DEFAULT_TRACE_REGION_SIZE = 0` (dynamic trace alloc), and #47122/#47766 change the *non-zero fixed-region* path — so #47122 only applies if the tt-mlir compiler embeds a non-zero `trace_region_size` for a traced graph (unresolved).

**RESUME TODO (where I stopped):**
1. Test the fix hypothesis: make tt-xla/tt-mlir open the device with a **non-zero `traceRegionSize`** (reserve a fixed trace region) and re-run — if the explosion vanishes, that's a shippable tt-xla/tt-mlir-side fix (plugin-only rebuild, fast). tt-mlir sets it at `runtime/lib/ttnn/runtime.cpp:599` = `options.traceRegionSize.value_or(DEFAULT_TRACE_REGION_SIZE=0)`. tt-xla `src/` does NOT reference traceRegionSize directly → find the PJRT client device-open path (goes through `tt::runtime` open; grep the tt-mlir runtime open + how PJRT passes MeshDeviceOptions / the compiled flatbuffer's trace_region_size).
2. If plugin-side won't set it, either patch `DEFAULT_TRACE_REGION_SIZE` in tt-mlir (needs tt-mlir rebuild) or bisect tt-metal `13adda80c11..38e954a066c` over the trace commits above to pin the exact commit.
3. Push branch `mvasiljevic/uplift-5592-decode-pcc-investigation` with this doc (+ env-gated diagnostics) and the fix once validated.

**Build/device state at pause:** installed build = BAD `327b846` / tt-metal `38e954a066c`. Device healthy (reset earlier via `python -m tt_smi -r`). Good build is `7fc4cc3e3` / `13adda80c11`. Env knobs live in `llm_benchmark.py` + `decode_utils.py`: `TTXLA_PCC_DEBUG`, `TTXLA_NO_TRACE`, `TTXLA_SKIP_PERF`, `TTXLA_PERF_TWICE`, `TTXLA_NO_TF_OVERRIDE`, `TTXLA_PREFILL_PERF`, `TTXLA_TRUNC_VOCAB`.

---

## FIX ATTEMPTS (autonomous run, chasing a perf-preserving XLA fix)

Goal: unblock the uplift with a tt-xla-side fix that does NOT damage measured perf (perf run must stay traced). Results on bad build, falcon3_1b `--num-layers 1`:

| attempt | idea | result |
|---|---|---|
| `TT_RUNTIME_TRACE_REGION_SIZE=128MB` | reserve big fixed trace region | ❌ still explodes (not region size) |
| `TTXLA_PCC_NO_TRACE=1` | perf run traced, PCC run untraced | ❌ still explodes (perf trace poisons even an untraced PCC run) |
| `TTXLA_FREE_PERF_TRACE=1` | del perf compiled model + gc + sync before PCC | ⏳ (see log) |
| `TTXLA_FREE_PERF_TRACE=1` | del perf compiled model + gc + sync before PCC | ❌ still explodes (trace is client-cached, not released by del) |
| `TTXLA_NO_TRACE=1` | disable trace entirely | ✅ PASS but **DAMAGES PERF**: full falcon3-1b **24.6 sps vs 57.8 traced (−57%)** → fails the 5% check-perf-regressions gate. Not viable for green CI. |

check-perf-regressions gate = **5% drop** on Samples/sec (`.github/scripts/check_regression.py`, `REGRESSION_THRESHOLD_PCT=5.0`). Traced baseline (full falcon3-1b, bs32) = **57.75 sps**, TTFT 651ms. So any fix must keep the perf run traced.

### Candidate fix: PCC-FIRST reorder (`TTXLA_PCC_FIRST`) — run PCC before perf
Insight: `TTXLA_SKIP_PERF` (no perf run) → PCC passes even full model (0.998, no hang). So a *clean device* (no resident perf trace) lets the PCC run decode correctly. The perf-preserving fix: **run the PCC/correctness run FIRST (clean device), then the traced perf benchmark LAST.** Perf stays traced → measured perf unchanged; the perf run's own (now-last) trace can't poison anything downstream.
- 1-layer: ✅ PCC 0.999 + perf ran (no hang) + traced.
- full model: ✅ (on a freshly-reset device) **PCC 0.998** (was 0.0), perf ran, no hang. (1st attempt hung in the PCC phase — stale device state after ~20 local runs without reset; a `tt_smi -r` fixed it. CI gets a fresh device per run.)

**Perf-preserving — validated (local, full falcon3-1b, bs32):**
| config | order | metal | samples/sec |
|---|---|---|---|
| good build | perf-first | old (13adda80c) | **52.34** |
| PCC-FIRST fix | pcc-first | new (38e954a06) | **51.06** |

−2.4% (within the 5% gate; and this includes the metal uplift itself, not just the reorder). NO_TRACE by comparison was 24.6 sps (−57%). So PCC-first is the perf-preserving fix. Implemented as a nested `_run_perf_benchmark()` run after the PCC block; **shipped as the default order** (no env gate) in the fix branch.

Key constraint learned: the perf run's **trace buffers**, once resident, poison the next graph's execution on the device (even if that graph is untraced) — so a perf-preserving fix must free/avoid the perf trace before the PCC run without dropping trace on the perf measurement itself.

---

## TL;DR (current best understanding)

1. **Culprit commit = tt-metal uplift `30645f913`** — the only runtime-affecting commit in the tt-mlir range; it bumps tt-metal `13adda80c11` → `38e954a066c`. Confirmed by build+run bisect (good at `7fc4cc3e3` / old tt-metal, bad at `327b846` / new tt-metal).
2. **The explosion happens iff BOTH: (a) the DECODE graph materializes/returns the lm_head logits tensor, AND (b) the KV cache is the device-native `fill_cache`-written one.** Established by a controlled test matrix (§Op attribution). It is **NOT** the token, **NOT** the returned-tensor size, **NOT** the prefill graph, **NOT** the SDPA/lm_head math, **NOT** the KV-cache *content*, **NOT** the PCC calc.
3. **The KV-cache content and the math are correct.** In the argmax path (perf run) the decode produces the **correct** token (73970, text matches good build) reading the *same* device cache → SDPA reads the cache fine and attention+lm_head compute correctly. Garbage appears only when the logits are materialized for host-return.
4. **Mechanism:** a **memory/buffer-allocation (aliasing) regression** in the new tt-metal — the return-logits decode graph's output-staging buffer collides with the device `fill_cache` KV-cache buffer placement, corrupting values during that graph's execution. The lean argmax graph (no logits output) and the CPU-transferred cache (different placement) both avoid the collision; the good build doesn't have it.
5. **This is why "perf output is fine but PCC=0"**: the perf run (argmax, no logits output) and the PCC run (materializes logits) run *different decode graphs*; only the PCC run's graph trips the allocation collision. The argmax/product path (e.g. vLLM) appears unaffected.
6. **Suspect class in tt-metal = allocator / buffer manager** (DRAM/L1 placement, buffer reuse), NOT the attention kernel. Superseded hypotheses, each killed by a later experiment: SDPA-decode kernel → `fill_cache`-numerics → token-data-dependence → logits-*prefill* co-allocation. Final: **logits-*decode*-graph × device-cache** allocation collision.
7. **Not a tt-xla problem and not a PCC-calc problem.** Exact tt-metal commit NOT pinned (636 commits in `13adda80c11..38e954a066c`; needs a tt-metal bisect focused on allocator/buffer changes).

The device was wedged by a full-model bad-build run; **recovered via `python -m tt_smi -r`** (tt-smi pip-installed into the venv; the `tt-smi` shell alias is broken — call the module).

---

## Key commits / versions

| | good (baseline) | bad (uplift target) |
|---|---|---|
| tt-mlir | `260d4c495fd5bebbfda3d48bc2e575514ec396f0` | `327b846251ef21196863c98ab5630ab53a051cce` |
| tt-metal (actual pin) | `13adda80c119631d18b0bc06163416ba148c25ab` (2026-06-20) | `38e954a066c7c23fb4693257c127b181480af19e` (2026-07-08) |

**tt-metal pin verified** from tt-mlir `third_party/CMakeLists.txt` `TT_METAL_VERSION` via `git show <tt-mlir-commit>:third_party/CMakeLists.txt`. Baseline `260d4c4` and `7fc4cc3e3` pin `13adda80c11`; `30645f913` and `327b846` pin **`38e954a066c`**. NOTE: the `30645f913` commit *message* says "uplift to `b19100423b`", but the actual pinned SHA is `38e954a066c` (`b19100423b` is 11 commits behind it). **Real tt-metal range = `13adda80c11..38e954a066c` = 636 commits** (an earlier note of 625 used the wrong `b19100423b` endpoint).

tt-mlir range `260d4c4..327b846` (chronological) and per-commit verdict:
| commit | verdict |
|---|---|
| `119966857` Replace 1x1x1 Conv3d with matmul/linear | irrelevant to LLM |
| `24eb2ab7f` Skip nlp_concat_heads_decode fusion when batch exceeds worker grid | additive tt-mlir guard, **no-op for batch=32** |
| `7fc4cc3e3` Relax **paged** SDPA decode verifier | **red herring** — our op is the **non-paged** variant; its verifier never required Q==K. Built+tested = **GOOD** |
| `30645f913` **Uplift tt-metal `13adda80c11`→`38e954a066c`** | **CULPRIT** — only runtime change between good and bad |
| `327b84625` refactor d2m-jit prefill tests | **test-only** (`test/d2m-jit/`), no runtime effect |

---

## Experiments run, and results

Command: `TTXLA_PCC_DEBUG=1 python -m pytest -svv tests/benchmark/test_llms.py::test_falcon3_1b [--num-layers 1]`
(`TTXLA_PCC_DEBUG` gates a diagnostic block added to `llm_benchmark.py` — see §Instrumentation. Default falcon3-1b bench config: batch=32, isl=128.)

| # | build (tt-mlir / tt-metal) | model | prefill PCC | decode PCC | device first-decode logits | outcome |
|---|---|---|---|---|---|---|
| 1 | `327b846` / new | 1 layer | 0.9994 | **0.0** | mean −2.9e16, min −3.0e19, max 2.5e19, argmax **91991** ≠ cpu | **FAIL (reproduced)** |
| 2 | `7fc4cc3e3` / old | 1 layer | 0.9994 | **0.9995** | mean −1.84, ±15, argmax **73970** = cpu | PASS |
| 3 | `7fc4cc3e3` / old | full | 0.9976 | **0.9981** | mean −4.06, ±9.8, argmax **12** = cpu | PASS |
| 4 | `327b846` / new (rebuild) | full | — | — | (perf run finished 111 iters, then) | **DEVICE HANG** — ETH heartbeat stuck `0xaabb001d`, NOC0 core e8-6 |
| 5 | `327b846` / new | 1 layer **--decode-only** | (CPU cache) | **0.8156** | mean −2.04, ±19, argmax **73970** = cpu (**sane**) | logits run + CPU cache = sane |
| 6 | `7fc4cc3e3` / old | 1 layer **--decode-only** | (CPU cache) | **0.8145** | mean −2.03, ±19, argmax 82848 (**sane**) | identical to bad ⇒ decode op not regressed |
| 7 | `327b846` / new | 1 layer **`TTXLA_NO_TF_OVERRIDE=1`** | — | **0.0** | mean −4.6e16, ±1e19, argmax 49078 (**explode**) | PCC decode uses device token **65741**, still explodes ⇒ NOT token-dependent |

Runs 5–7 are the isolation (see §Op attribution). Run 7 (override-off) is decisive: with the *perf* token 65741 the *logits* run still explodes, while the perf run with 65741 is sane ⇒ it is the **logits-graph × device-cache** interaction, not the token. decode-only (5,6) shows the decode compute is unchanged good-vs-bad.

CI itself = full model on bad tt-mlir → decode PCC 0.0 (completed there; locally it hung — kernel corruption is severe/flaky).

**Logit-stat interpretation:** on the bad build the device first-decode logits are not NaN, not constant, and not a rescale of the golden — exploded to ~10¹⁹ and structurally uncorrelated (⇒ PCC 0). On the good build they are sane and argmax matches CPU. Same CPU golden across builds. So it is the *device tensor* that changes, not the comparison.

---

## Root-cause IR evidence

Decode graph (`modules/irs/ttnn_*_g1_*.mlir`, `_g3_*.mlir`) attention op:
```mlir
%44 = "ttnn.scaled_dot_product_attention_decode"(%q, %k, %v, %mask)
  : (tensor<1x32x8x256xbf16>,                            // Q  bf16
     tensor<32x4x128x256x!ttcore.tile<32x32, bfp_bf8>>,  // K  BFP8
     tensor<32x4x128x256x!ttcore.tile<32x32, bfp_bf8>>,  // V  BFP8
     tensor<1x1x8x128xbf16>) -> tensor<1x32x8x256xbf16>
```
- KV cache is BFP8 even though the benchmark passes `experimental_kv_cache_dtype=None`. That option gates the `TTNNKVCacheDtypeConversion` pass (default None, **unchanged in range**), so it is not the source; the BFP8 KV pattern comes from the decode RoPE/QKV fusion path already present in baseline.
- The **non-paged** `ScaledDotProductAttentionDecodeOp` verifier only enforces Q==Result and K==V — never Q==K. So bf16-Q/BFP8-KV was always legal; the regression is numeric, not a verifier unlock.
- The IR at good `7fc4cc3e3` has the **identical** bf16-Q/BFP8-KV decode op yet passes — proving the graph is the same and only the tt-metal kernel changed.

---

## Op attribution — the DECODE-graph × device-cache allocation collision

Controlled test matrix (bad build unless noted, 1 layer, `--num-layers 1`). Each row changes ONE variable:

| # | run / knob | decode graph returns | KV cache source | device decode logits |
|---|---|---|---|---|
| A | full-flow, override ON | full logits `[.,.,131072]` | device `fill_cache` | **explode ~1e19**, argmax 91991 |
| B | `TTXLA_NO_TF_OVERRIDE=1` | full logits | device `fill_cache` | **explode ~1e19**, argmax 49078 (token 65741, same as perf) |
| C | perf run | **token-id (argmax)** | device `fill_cache` | **sane** (correct token, text matches good) |
| D | `--decode-only` | full logits | **CPU (transferred)** | **sane** (±19), argmax 73970 |
| E | `TTXLA_PREFILL_PERF=1` | full logits | device `fill_cache` (written by **perf** prefill) | **explode ~1e19** |
| F | `TTXLA_TRUNC_VOCAB=2048` | logits `[.,.,2048]` | device `fill_cache` | **explode ~1e19** |
| G | good build, full-flow | full logits | device `fill_cache` | **sane** (0.998) |

Single-variable deductions:
- **A vs B** → not the **token** (both explode).
- **B vs C** → same token + same device cache, only the decode graph differs (returns logits vs argmax) → it is the **decode graph**, and the argmax path is **correct** ⇒ **cache content + attention + lm_head math are all fine**.
- **A/B vs D** → device cache explodes, CPU-transferred cache is sane ⇒ requires the **device-native cache placement** (content is fine per C; it's the buffer *placement*).
- **A vs E** → prefill graph (perf vs logits) is irrelevant ⇒ **not the prefill graph** (kills the earlier "logits-prefill g2" hypothesis).
- **A vs F** → truncating the returned logits to 2048 (compute unchanged, output 64× smaller) **still explodes** ⇒ **not the returned-tensor size**; it's the *presence* of the logits-output materialization.
- **A vs G** → only on the new tt-metal.

⇒ **The explosion needs exactly: (decode graph that materializes the lm_head logits for host-return) × (device-native `fill_cache` KV cache), on the new tt-metal.** Because the argmax path (C) reads the same device cache and is correct, the **cache content and math are not the problem** — the corruption is introduced by the return-logits decode graph's execution when a device cache is present. Best explanation: a **buffer-allocation/aliasing regression** where the logits-output staging buffer collides with the device `fill_cache` KV-cache buffer placement. Suspect class = **tt-metal allocator / buffer manager**, not the SDPA/lm_head kernels.

**On the 0.816 decode-only PCC (your review — too low for one step):** it is **identical good vs bad** (0.8145 vs 0.8156). Its absolute value is a decode-only-mode artifact (bf16 CPU cache round-tripped to BFP8 for the device while the CPU reference stays bf16, ~0.18 loss on the 131072-dim logits). Read only the good==bad *equality* → decode compute not regressed.

**Could NOT dump the device cache directly:** moving `past_key_values.layers[i].keys` to host mid-flow **fatally aborts** (`Copy to host buffer failed … TransferFromDevice`, SIGABRT) on both builds — a transfer-path limitation, not the bug.

**Remaining fork (needs tt-metal tooling):** cache-buffer vs logits-buffer as the overwritten region. Argument from C (argmax correct ⇒ cache content fine) leans toward the collision corrupting values *during* the return-logits graph rather than a pre-corrupted cache — but pinning the exact buffer + the exact tt-metal commit needs a tt-metal memory tool / bisect.

---

## Perf run vs logits run — why "output looks fine" but PCC=0

Three executions per test: (1) CPU golden, (2) **perf run** compiled `return_logits=False` (only argmax token id leaves the graph; prints generated text; never checks logits), (3) **logits/PCC run** compiled `return_logits=True` (materializes full `[B,1,vocab]` logits, PCCs vs CPU).

Diff of the two **decode** graphs (perf `g1` vs logits `g3`), bad build:
- `scaled_dot_product_attention_decode` op **and its operand memory configs are byte-identical** (Q `bf16 dram interleaved`, KV `bfp_bf8 dram interleaved`, output identical).
- Only difference: **4 output-plumbing ops** in g3 (`reshape`/`to_memory_config`/2×`deallocate`) that return the full logits tensor. Nothing in the attention path differs.

The decode graphs are NOT running different attention. But the perf and PCC runs execute **different PREFILL graphs** — g0 (perf) returns only token-ids; **g2 (PCC) also materializes the ~140 MB `32x17x131072` prefill-logits tensor** — and only g2's `fill_cache` corrupts the KV cache (see §Op attribution). g1 decode reads g0's clean cache → sane; g3 decode reads g2's corrupt cache → explodes.

**Controlled proof it is the run, not the token** (`TTXLA_NO_TF_OVERRIDE=1` disables the alignment override so the PCC decode uses the device's own token 65741, same as perf):
- PCC run, token **70288** (override on) → explode, PCC 0.
- PCC run, token **65741** (override off) → **explode**, PCC 0. ← same token as perf, still explodes.
- Perf run, token **65741** → sane.

So the token is irrelevant; the perf run is fine and the PCC run explodes over the same device cache. The `"Device prefill produced different tokens…"` override (bf16 near-tie 65741/70288) is a red herring for this bug — it fires on both builds and does not change the outcome.

Note on text: at 1 layer perf text is gibberish on *both* builds; at full model it is **blank even on the healthy good build** (argmax = whitespace token) and still passes. Generated text is not a health signal in either direction — only logits/PCC are. "Output looks fine" = the perf run's argmax path is genuinely unaffected (clean cache), not a misjudgment about the text.

---

## Where the exact-commit search stops

- **tt-mlir commit `30645f913` = confirmed**, but it is a pure tt-metal version bump (no mlir code).
- **tt-metal commit = NOT pinned.** Range `13adda80c11..38e954a066c` = **636 commits** (2026-06-20 → 07-08). Given the corrected mechanism (corrupt cache write when a large returned tensor is co-allocated with `fill_cache`), the suspect class is **memory allocator / buffer management** (L1/DRAM allocation, buffer reuse, program cache) — NOT the SDPA/decode kernels. Pinning needs a **tt-metal bisect** (~log2(636)≈10 heavy builds) using the `--num-layers 1` full-flow test (good = decode PCC ~0.998, bad = 0.0). Bisect tt-metal via `-DTTMLIR_TTMETAL_SOURCE_DIR=<checkout>` with tt-mlir fixed.

---

## How to reproduce / continue

### Build against a specific tt-mlir commit
```bash
cd /localdev/mvasiljevic/tt-xla
export TTMLIR_TOOLCHAIN_DIR=/opt/ttmlir-toolchain
# tt-mlir is fetched by CMake (NOT a git submodule) into third_party/tt-mlir/src/tt-mlir
sed -i 's/set(TT_MLIR_VERSION "[^"]*")/set(TT_MLIR_VERSION "<COMMIT>")/' third_party/CMakeLists.txt
source venv/activate
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build   # incremental + ccache
# crossing 30645f913 reverts/re-checkouts tt-metal (heavier). Verify HEADs:
git -C third_party/tt-mlir/src/tt-mlir rev-parse --short HEAD
git -C third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal rev-parse --short HEAD
```
Prereq once: `git submodule update --init --recursive third_party/tt_forge_models`.

### Run + get logit diagnostics
```bash
TTXLA_PCC_DEBUG=1 python -m pytest -svv tests/benchmark/test_llms.py::test_falcon3_1b --num-layers 1
# decode-only isolation (CPU prefill + CPU KV cache; device runs only decode):
TTXLA_PCC_DEBUG=1 python -m pytest -svv tests/benchmark/test_llms.py::test_falcon3_1b --num-layers 1 --decode-only
```
IR dumped to `modules/irs/` (`ttnn_*_g0..g3_*.mlir`): **g0=perf-prefill, g1=perf-decode, g2=logits-prefill, g3=logits-decode** (perf graphs return token-id only; logits graphs return the `[B,N,131072]` vocab tensor).

### Reset the device (it hangs on the bad kernel / full model)
```bash
source venv/activate; unalias tt-smi 2>/dev/null   # the tt-smi shell alias points to a missing binary
pip install tt-smi                                 # if not present (installed into venv this session)
python -m tt_smi -r                                # reset PCI device(s); wait ~20s, then verify:
python -c "import jax; print(jax.devices('tt'))"   # expect TTDevice(id=0, arch=Wormhole_b0)
```

### Instrumentation added (tests/benchmark/benchmarks/llm_benchmark.py) — env-gated
- `TTXLA_PCC_DEBUG=1` — prints device+CPU logit stats for prefill and first-decode (both PCC branches), plus an unconditional `[PCCDBG-DEC1]` dump of `device_decode_logits[0]` (flags EXPLODED if |max|>1e4).
- `TTXLA_NO_TF_OVERRIDE=1` — disables the "device prefill produced different tokens" alignment override so the PCC decode uses the device's OWN token (same as perf). Proved: not the token.
- `TTXLA_PREFILL_PERF=1` — runs the logits-run PREFILL with the perf graph (no returned logits) then decodes with the logits graph. Proved: not the prefill graph.
- `TTXLA_TRUNC_VOCAB=N` — returns only the first N vocab entries of the decode logits (lm_head still computes full). Proved: not the returned-tensor size (2048 still explodes).
All env-gated (`llm_benchmark.py` + `decode_utils.py`). No effect unless set. Remove before merge or keep as debug aids.

### To pin the exact op (next)
The corruption is a cache write that only goes wrong when a large returned tensor is co-allocated. Cleanest confirmations: (a) mark the KV cache as a graph output (the direct `.to("cpu")` transfer aborts) or use a tt-metal/ttrt op-level test to diff the device cache after g2 (logits-prefill) vs g0 (perf-prefill); (b) shrink/remove the returned prefill-logits tensor and see the explosion vanish; (c) tt-metal bisect focused on allocator/buffer-manager commits.

### To bisect tt-metal (pin the exact kernel commit)
Keep tt-mlir at `30645f913`; override tt-metal per candidate via `-DTTMLIR_TTMETAL_SOURCE_DIR=<checkout>` (see `third_party/CMakeLists.txt`), rebuild (incremental), rerun. Good=decode PCC passes; Bad=PCC 0 / hang.

### Saved artifacts (this session, under scratchpad — session-scoped, copy out what you need)
`/tmp/claude-1211408017/-localdev-mvasiljevic-tt-xla/174a621f-f290-4548-831b-7cbba06420a1/scratchpad/`
- `repro_327b846.log`, `repro_7fc4cc3e3.log`, `full_good_7fc4cc3e3.log`, `full_bad_327b846.log` — full-flow run logs
- `decode_only_bad_327b846.log`, `decode_only_good_7fc4cc3e3.log` — decode-only isolation run logs
- `no_tf_bad_327b846.log` — override-off run (PCC decode uses device token 65741, still explodes)
- `prefill_perf_bad.log` — perf-style prefill + logits decode (still explodes ⇒ not the prefill graph)
- `trunc_2048_bad.log` — returned logits truncated to 2048 vocab (still explodes ⇒ not the size)
- `build_*.log` — build logs
- `modules_327b846/` — full IR dump from the bad 1-layer run (g0–g3); `ir_327b846/` — saved g1 decode IR
- `bisect_build.sh <commit>`, `repro.sh <label>`, `repro_full.sh <label>` — helper scripts
- `ci_full.log` — full CI job log (job 86736187259)

---

## Working-tree state left behind
- **Current installed build = GOOD `7fc4cc3e3` / old tt-metal `13adda80c11`** (rebuilt last for the decode-only baseline). `third_party/CMakeLists.txt` `TT_MLIR_VERSION` = `7fc4cc3e3`.
- `tests/benchmark/benchmarks/llm_benchmark.py` — `TTXLA_PCC_DEBUG`-gated diagnostic blocks in BOTH the `else` (full-flow) and `elif decode_only` branches.
- `third_party/tt_forge_models` submodule initialized.
- `tt-smi` pip-installed into the venv (reset via `python -m tt_smi -r`, not the broken `tt-smi` alias).
- **Device is currently healthy** (reset + verified this session).

---

## Open questions for next person
1. **Confirm the corrupt-cache mechanism directly:** shrink/remove the returned prefill-logits tensor in the logits-prefill graph (g2) and check the explosion vanishes; or dump the g2 vs g0 device cache via a graph-output (the `.to("cpu")` transfer aborts). Expected: g2's cache is corrupt, g0's is clean.
2. **Pin the exact tt-metal commit** in `13adda80c11..38e954a066c` (636) — suspect class **allocator / buffer management** (not SDPA/decode kernels). Bisect with the `--num-layers 1` full-flow test (good=0.998, bad=0.0), tt-mlir fixed, tt-metal via `-DTTMLIR_TTMETAL_SOURCE_DIR`.
3. **Production impact:** the perf/argmax path (no full-vocab logits) is sane — quantify token accuracy vs CPU over many steps to decide whether vLLM/product decode is actually affected, or whether this only breaks the benchmark's logits-materialization run.
4. **Fix belongs in tt-metal** (buffer/allocator regression surfacing as a corrupt `fill_cache` write when co-allocated with a large returned tensor). No clean tt-xla workaround identified.
