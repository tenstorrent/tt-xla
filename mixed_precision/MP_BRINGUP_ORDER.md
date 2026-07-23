# Mixed Precision (MP) Bringup Order — vLLM benchmarks

Working plan for the agentic MP bringup effort. Ordered so that the models we
can run **right now on n150** come first. This is a living document — the order
and per-model notes will be refined as models are actually brought up.

## The three MP knobs

| Knob | vLLM config field | Compiler option | Default (unset) | Values |
|---|---|---|---|---|
| Weight dtype (global) | `experimental_weight_dtype` | `experimental_weight_dtype` | `""` = **bf16** | `bfp_bf8`, `bfp_bf4` |
| Weight dtype (per-tensor) | `weight_dtype_overrides` | host-side `tt.weight_dtype_override` custom_call | `None` | `{glob: dtype}` + `"default"` key; overrides the global dtype |
| KV cache dtype | `experimental_kv_cache_dtype` | `experimental-kv-cache-dtype` | `None` = **bf16** | `bfp_bf8` ✅ · `bfp_bf4` ❌ (not impl, tt-xla #5011) |
| Activation dtype lowering | `enable_activation_dtype_lowering` | `enable_activation_dtype_lowering` | `false` | `true` (lowers activations to bfp8 around CCL ops; Llama-style O-proj/MLP sub-graphs) |

**Target MP recipe** for a fully-tuned decode LLM:
- Weights: MLPs → `bfp_bf4`, everything else → `bfp_bf8` (via `weight_dtype_overrides`)
- KV cache → `bfp_bf8`
- Activations → lowered to bfp8 around CCL ops (`enable_activation_dtype_lowering=True`, mostly relevant on multi-device TP)

Each step is gated on accuracy not regressing, measured with the vLLM
teacher-forced accuracy harness (cherry-picked from PR #5615 onto this branch
until it lands on main).

> **Note on `enable_activation_dtype_lowering`:** originally unreachable from the
> vLLM path (no `TTConfig` field). Plumbed into `TTConfig` /
> `get_pjrt_compile_config()` on this branch. Most impactful on multi-device TP
> models where it cuts bytes moved by collectives; low/no benefit single-chip.

## Current state (before bringup)

Machine mapping (from `.github/workflows/perf-bench-matrix.json`):
- **All single-device vLLM configs → n150-perf / p150-perf.** ← we are here.
- All TP configs → qb2-blackhole (Mistral-Small-3.1 → galaxy-wh-6u).

On n150 today:
- Decode LLMs are all at **Tier 1**: weight `bfp_bf8` (global), KV `bf16`, no per-op bfp4, no activation lowering.
- Embedding models are at **Tier 0**: weight `bf16` (only the DP variants are already `bfp_bf8`).
- There are **no untuned decode LLMs on n150** — the fully-untuned (bf16-weight) decode models are all TP on qb2/galaxy.

---

## Phase 1 — n150 (current machine)

### 1a. Embeddings (Tier 0 → weight bfp8)

Only knob that applies to encoder/embedding models is weight dtype (no KV cache,
no autoregressive decode). Quick warm-up targets; validate embedding-quality
metric, not token accuracy.

| Test id | Model | Now | Target |
|---|---|---|---|
| `bge-m3-batch1` | BAAI/bge-m3 | bf16 | weight `bfp_bf8` |
| `bge-m3-batch32` | BAAI/bge-m3 | bf16 | weight `bfp_bf8` |
| `qwen3-embedding-4b-batch1` | Qwen3-Embedding-4B | bf16 | weight `bfp_bf8` |

(The `*-dp-batch32` variants of these already run `bfp_bf8` — use them as the reference config.)

### 1b. Decode LLMs (Tier 1 → add KV bfp8 → bfp4 MLPs → activation)

All below currently: weight `bfp_bf8` global, KV `bf16`, no per-op override, no
activation lowering. Per model, apply in order and re-check accuracy after each:

1. `experimental_kv_cache_dtype="bfp_bf8"`
2. `weight_dtype_overrides={"default":"bfp_bf8", "model.layers.*.mlp.*proj":"bfp_bf4"}` (MLPs → bfp4)
3. `enable_activation_dtype_lowering=True` (expect little single-chip gain; validate it doesn't regress accuracy)

Ordered smallest → largest (fast iteration, easier accuracy debugging on small models first):

| # | Test id | Model | Params |
|---|---|---|---|
| 1 | `qwen2.5-0.5b-instruct` | Qwen2.5-0.5B-Instruct | 0.5B |
| 2 | `qwen3-0.6b` | Qwen3-0.6B | 0.6B |
| 3 | `falcon3-1b-base` | Falcon3-1B-Base | 1B |
| 4 | `llama-3.2-1b` | Llama-3.2-1B-Instruct | 1B |
| 5 | `phi-1` | microsoft/phi-1 | 1.3B |
| 6 | `phi-1_5` | microsoft/phi-1_5 | 1.3B |
| 7 | `qwen2.5-1.5b-instruct` | Qwen2.5-1.5B-Instruct | 1.5B |
| 8 | `qwen3-1.7b` | Qwen3-1.7B | 1.7B |
| 9 | `gemma-1.1-2b-it` | google/gemma-1.1-2b-it | 2B |
| 10 | `phi-2` | microsoft/phi-2 | 2.7B |
| 11 | `qwen2.5-3b-instruct` | Qwen2.5-3B-Instruct | 3B |
| 12 | `llama-3.2-3b` | Llama-3.2-3B-Instruct | 3B |
| 13 | `falcon3-3b-base` | Falcon3-3B-Base | 3B |
| 14 | `qwen3-4b` | Qwen3-4B | 4B |
| 15 | `falcon3-7b-base` | Falcon3-7B-Base | 7B |
| 16 | `mistral-7b-instruct` | Mistral-7B-Instruct-v0.3 | 7B |
| 17 | `qwen2.5-7b-instruct` | Qwen2.5-7B-Instruct | 7B |
| 18 | `qwen3-8b` | Qwen3-8B | 8B |
| 19 | `llama-3.1-8b` | Llama-3.1-8B-Instruct | 8B |
| 20 | `ministral-8b` | Ministral-8B-Instruct-2410 | 8B |

Canary: `opt-125m` (vLLM-only, not in the perf matrix) — useful smoke test.

---

## Phase 2 — qb2-blackhole / galaxy (deferred until on that HW)

These are the **most untuned** models (weight still `bf16`), so the largest raw
win — but they need multi-device hardware. Bring up: weight `bf16 → bfp_bf8` →
KV `bfp_bf8` → bfp4 MLPs → `enable_activation_dtype_lowering=True` (this is where
activation lowering actually pays off, per the torch-xla Llama-3.1-70B result).

Tier 0 (weight bf16), qb2-blackhole:
- `qwen3-32b-qb2-tp` — Qwen3-32B
- `falcon3-7b-qb2-tp` — Falcon3-7B-Base
- `falcon3-10b-qb2-tp` — Falcon3-10B-Base
- `qwen2.5-coder-32b-instruct-qb2-tp` — Qwen2.5-Coder-32B
- `mistral-small-24b-instruct-2501-qb2-tp` — Mistral-Small-24B-2501
- `llama-3.1-8b-qb2-tp` — Llama-3.1-8B
- `gemma4-31b-it-tp` — Gemma-4-31B-it (text-only)

Partially tuned (weight bfp8 already), finish KV / bfp4 / activation:
- `llama-3.1-70b-qb2-tp` — weight bfp8, KV bf16
- `mistral-small-3.1-24b-tp` (galaxy) — weight bfp8 **+ KV bfp8** already; add bfp4 MLPs / activation
