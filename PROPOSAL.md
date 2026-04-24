# Proposal: Porting tt-quetzalcoatlus Optimization Patterns into TT-XLA / TT-MLIR

## Context

This branch (`nkapre/quetzal-fx-rewrites`) added an FX-level pre-pass to TT-XLA
that lifts selected quetzalcoatlus-inspired rewrites (tanh-GELU reconstruction,
manual-SDPA reconstruction) into the torch.compile path so they appear as
`tenstorrent.*` composites in StableHLO. The infrastructure is working: 8/8
focused unit tests pass, the IR-comparison script produces the expected
StableHLO/TTIR/TTNN deltas on toy graphs, and end-to-end validation against
8× Wormhole_b0 hardware produces correct PCC on every model that compiles.

However, after running a three-way A/B/C comparison (see
`scripts/compare_three_way.py`) on real HuggingFace models, the runtime
evidence is stark: **the FX pre-pass, as currently scoped, does not produce a
measurable perf benefit on any tested model.**

This document proposes how to close that gap — not by changing the pre-pass,
but by landing, one-by-one, the specific tt-mlir and tt-xla changes that
together replicate what tt-quetzalcoatlus does.

## Observed data

Three-way comparison on 8× Wormhole_b0, layer-only, tracy-instrumented tt-metal
trace replay (`enable_trace=True`), min of N runs:

| Model | A (tt-xla native) | B (tt-xla + quetzal pre-pass) | C (quetzal export→ttnn codegen) | PCC A/B/C |
|---|---:|---:|---:|---|
| Llama-3.2-1B | 1.466 ms | 1.493 ms (+1.8%) | 1.259 ms (−14%) | 0.999993 / 0.999993 / 0.999990 |
| Llama-3.1-8B | 2.476 ms | 2.492 ms (+0.6%) | 4.536 ms (+83%)  | 0.999992 / 0.999992 / 0.999973 |
| GPT-2 | 1.501 ms | 1.558 ms (+3.8%) | 0.670 ms (−55%)  | 0.999912 / 0.999912 / 0.989973 |
| bert-base-uncased | tt-mlir crash (pre-existing, both A and B) | same crash | 0.662 ms | — / — / 0.948860 |

Two distinct signals from this table:

1. **Pre-pass B vs A is within noise**, with one measurable regression (GPT-2
   after the new dynamo-form pattern fires: +3.8%). The IR has 18 new
   `tenstorrent.gelu_tanh` composites but this does not translate to runtime
   savings because tt-mlir was already reclaiming the same `ttnn.gelu` op
   downstream before the pre-pass marked it explicitly.

2. **C vs A is inconsistent**: C is 2.2× faster on GPT-2, 1.16× faster on
   Llama-1B, but 1.8× *slower* on Llama-8B. So the quetzal codegen path has
   no uniform advantage — it has specific wins on specific shapes.

Conclusion: the interesting question is not "why is the pre-pass not helping"
(answer: it can only affect StableHLO-level composite boundaries, and tt-mlir
already handles those), but **"what does quetzal's generated.py do below the
StableHLO layer that tt-xla's path does not — and where does each of those
fixes need to land in the Tenstorrent stack?"**

## Quetzal's wins, decomposed by stack layer

Inventory based on inspection of `generated.py` outputs and
`metadata.json` fusion reports from tt-quetzalcoatlus:

| Optimization | What it does | Stack layer where it lives | Can FX pre-pass do it? |
|---|---|---|---|
| `ttnn.linear(activation="gelu"/"silu")` | Fuse matmul + unary activation into a single TTNN kernel | **tt-mlir TTIR→TTNN** | No |
| L1 vs DRAM memory config per op | Hand-pick intermediate residency | **tt-mlir layout/memory pass** | No |
| Tile-layout stability | Tilize once, stay tiled; avoid round-trips | **tt-mlir layout propagation** | No |
| `WormholeComputeKernelConfig(HiFi3, fp32_acc)` | Math fidelity + accumulation mode per op | **tt-mlir TTNN emitter** | No |
| `fuse_gate_up` (gate_proj + up_proj → concat-linear) | One wider matmul + split | FX pre-pass (needs torch_xla consteval to fold weight cat) | Yes, conditionally |
| `fuse_qkv_proj` (Q+K+V → concat-linear + split) | Same idea for attention input | FX pre-pass (same condition) | Yes, conditionally |
| `fuse_rope` (rotary embedding reclamation) | Match decomposed RoPE, emit rotary composite | FX pre-pass + **tt-mlir lowering + `ttnn.rotary_embedding` op** | Only paired with tt-mlir work |
| `fuse_rmsnorm` | Reconstruct decomposed `pow→mean→add→rsqrt→mul→mul` | FX pre-pass | **Already done** (`RMSNormFusionProvider`, default-on) |
| `fuse_gelu` (tanh variant) | Reconstruct decomposed tanh-GELU | FX pre-pass | **Already done** (this branch; dynamo-form pattern added today) |
| `fuse_sdpa` | Reconstruct manual matmul+softmax+matmul | FX pre-pass | **Already done** (this branch, confirmed matches dynamo graphs) |
| No-StableHLO codegen | Skip the whole MLIR stack | Architectural | No — TT-XLA is a PJRT backend by design |

So of the ~10 patterns quetzal applies, 3 are already in the FX pre-pass, 2 more
are FX-pre-pass-achievable but conditional on consteval behavior and model
shape, and 5 live below the StableHLO boundary and are inherently tt-mlir
work.

## Proposed changes, ranked by ROI / effort

### Tier 1 — tt-mlir, highest ROI

**1A. `matmul + unary_activation → ttnn.linear(activation=...)`**

- Scope: tt-mlir TTIR-to-TTNN conversion pattern. Match `ttnn.matmul` immediately
  followed by `ttnn.gelu` / `ttnn.silu` / `ttnn.relu` where the activation's
  only user is outside the matmul's consumers, rewrite to a single
  `ttnn.linear` with `activation=` set.
- Effort: ~1-2 weeks for a single contributor.
- Expected impact: closes most of the GPT-2 C-vs-A gap immediately (the c_fc
  projection — 768→3072 — runs as one kernel in quetzal, two in tt-xla). Helps
  every transformer MLP with a unary activation.
- Regression test: `scripts/compare_three_way.py openai-community/gpt2` must
  show B's runtime collapse toward C's.
- PR target: `tenstorrent/tt-mlir`, under `lib/Conversion/TTIRToTTNN/` or the
  equivalent TTNN-dialect optimization passes.

### Tier 2 — tt-mlir, medium ROI

**2A. `tenstorrent.rotary_embedding` composite + lowering**

- Scope: three coordinated pieces.
  1. In this tt-xla branch: add a provider that matches the decomposed RoPE
     pattern (`[mul, getitem, neg, cat, mul, add]` — see QUETZAL_PLAN.md) and
     wraps it in a `stablehlo.composite "tenstorrent.rotary_embedding"` via
     `StableHLOCompositeBuilder`.
  2. In tt-mlir: add the StableHLO→TTIR lowering for the composite; emit
     whichever TTIR rotary op corresponds.
  3. Verify the TTIR→TTNN path emits `ttnn.experimental.rotary_embedding_llama`
     (or equivalent). If the TTNN op does not exist, add it in tt-metal first.
- Effort: 1-2 weeks for the lowering; Python side is ~20 lines.
- Impact: broad — every Llama / Mistral / Qwen / Phi family model has RoPE
  per attention layer.
- PR target: paired tt-mlir + tt-xla + possibly tt-metal.

**2B. Memory-config heuristic**

- Scope: tt-mlir pass that estimates per-op L1 footprint during TTIR→TTNN
  lowering and sets `memory_config=L1_MEMORY_CONFIG` for intermediates that fit,
  DRAM otherwise. Quetzal hardcodes this per-op; tt-mlir's current default is
  conservative (DRAM for everything).
- Effort: multi-week, needs a simple cost model and tests across many shapes to
  avoid regressions.
- Impact: shape-dependent; biggest wins on small-intermediate ops (attention
  outputs, activation post-processing).
- PR target: tt-mlir, worth a design doc first.

### Tier 3 — tt-xla FX pre-pass, conditional

**3A. `fuse_gate_up` and `fuse_qkv_proj`**

- Scope: add FX providers that match `gate_proj + silu + up_proj → combined
  matmul + split` and the analogous QKV fusion. Replacement produces
  `F.linear(x, torch.cat([W_gate, W_up], dim=0), torch.cat([b_gate, b_up]))`
  then `torch.split(...)` back into two tensors. Rely on torch_xla consteval
  to fold the `torch.cat` out at compile time.
- Effort: 2-3 days per provider.
- Impact: **conditional**. Quetzal itself shows `fuse_gate_up` making Llama-8B
  *slower* (4.54ms C vs 2.48ms A). Likely helpful only below some hidden-dim
  threshold. Add a shape-based `match_filter` to gate the rewrite.
- Caveat worth verifying first: does torch_xla actually consteval the
  `torch.cat(W_gate, W_up)` out of the runtime graph when the weights are
  marked as parameters? Needs a one-afternoon test on a synthetic layer.
- PR target: follow-up on this tt-xla branch.

**3B. Add dynamo-form patterns to existing providers where applicable**

- Scope: every provider in `fusion_providers.py` should have at least one
  pattern variant that uses the node shapes dynamo produces (call_method
  for binary operators on Tensor Proxies, call_function for `torch.*` named
  calls). The GELU provider was missing its dynamo form until today's fix
  (`pattern_method_function_pow`). The RMSNorm and SDPA providers happen to
  work already — but other future providers will need this care.
- Effort: incremental, one pattern at a time as new providers are added.

### Tier 4 — Infrastructure improvements (lower priority)

**4A. Per-kernel tracy CSV capture in `compare_three_way.py`**

- Currently reports only wall-clock min-of-N trace-replay time. Useful, but
  obscures where time goes within the replay. Add an option to set
  `TT_METAL_PROFILE=1` and parse the emitted `ops_perf_results_*.csv` so the
  output table shows per-kernel timing attribution per variant. This would
  make future tt-mlir PRs easier to evaluate on a target basis.
- Effort: 1-2 days.

**4B. Cross-model sweep script**

- Generalize from the ~4 models we ran manually to a sweep over the 200
  models that quetzalcoatlus already has cached (`models_200.txt`), producing
  a summary table of A/B/C per-model. Would make regressions and wins much
  easier to spot.
- Effort: 1-2 days.

## Concrete path forward

1. **Land the activation-fusion tt-mlir PR** (1A above) as the single highest
   ROI next step. This does not require any of the other changes; it's a
   self-contained TTNN dialect rewrite. Target: a measurable ≥20% speedup on
   GPT-2 B via `compare_three_way.py`.

2. **Land the rotary embedding pair** (2A). This requires the FX side
   (trivial) + a tt-mlir lowering (moderate). Measures success as the first
   rewrite that is observably useful *because* of the pre-pass, not in spite
   of it.

3. **Benchmark `fuse_gate_up` and `fuse_qkv_proj`** (3A) conditionally on
   shape, before landing. Needs the torch_xla consteval check first.

4. **Park memory-config heuristic** (2B) behind a design doc — too many
   shapes to tune without a plan.

5. **Leave this branch as-is** for now. It is correct scaffolding. Future
   composite-based rewrites (RoPE, fused QKV, etc.) will ride on its
   `FusionProvider` registry and `tt_quetzal_rewrite_passes` option. Do not
   merge this branch as a perf improvement — the data doesn't support that
   framing. Merge it as infrastructure once at least one future rewrite
   with a verified win is ready to land alongside it.

## Coordination

The tt-quetzalcoatlus team has already done the pattern identification
work. Their `fusion_specs.yaml` file is a specification of which patterns
pay off on which shapes, and `metadata.json` in every compiled model's
directory is a record of which fusions fired. Both are a cheat sheet for
this effort. Before writing any tt-mlir PR, walk through quetzal's
compiled output for the target model family and confirm the pattern is
one quetzal actually exploited.

## Appendix: how to reproduce the numbers

```bash
export XLA_PY=/localdev/nkapre/venvs/xla/bin/python
export QUETZAL_DIR=/localdev/nkapre/tt-quetzalcoatlus
# Per docs/TT_XLA_BUILD_OPTIONS.md, set LD_LIBRARY_PATH appropriately.
python scripts/compare_three_way.py meta-llama/Llama-3.1-8B \
  --seq-len 1 --batch-size 1 --n-runs 5
```

Artifacts (per-variant stdout/stderr, snapshotted MLIR dumps, summary JSON)
land under `--output-dir` (default `/tmp/quetzal-threeway/`). The
`summary.json` in each run directory is the canonical numeric output.

To see whether the rewrite actually fired in variant B, set
`TTXLA_LOGGER_LEVEL=INFO` and look for
`[QuetzalRewrite] passes=... matches=N` lines in the variant-B stderr log.
`matches=0` means the patterns in `fusion_providers.py` do not match the
graph that dynamo handed to tt-xla on that model — which, for any model
using `torch.nn.functional.gelu` and `F.scaled_dot_product_attention`
directly, is the expected outcome today.
