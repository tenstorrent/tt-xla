# GPT-OSS quantization investigation — overview

This directory and a handful of helper scripts make up a self-contained
investigation into the **bfp4/bfp8 quantization accuracy** of GPT-OSS matmuls on
Tenstorrent hardware. The goal is to isolate individual matmul ops (and whole
transformer layers) from `openai/gpt-oss-120b`, run them on a TT device under a
full matrix of compiler configs, and measure the `relative_l2` (and `pcc`)
error before/after a compiler change.

> **This work is a backup / PoC snapshot — it is not intended to be merged into
> `main` as-is.** It is preserved here so the approach, the run commands, and the
> measured results are all reproducible later.

For the full design rationale and the **measured results** (including the
const-eval host-cast fix that gave ~40% rel_l2 reduction on bfp4), see
[`results.md`](results.md). This file is the entry point: what each piece is and
how to run it.

---

## File index

| File | Kind | Purpose |
|---|---|---|
| `test_matmul_gpt_oss_120b.py` | test | Isolated matmuls (q/k/v/o_proj, router, attn_score, attn_context) using **real** extracted GPT-OSS weights. 3 layers × 7 ops × 48 configs = **1008** cases. |
| `test_matmul_gpt_oss_120b_random.py` | test | Same shape matrix, but **random** weights for every op. Used to check whether the bfp4 issue reproduces without real model weights. |
| `test_layer_gpt_oss_120b.py` | test | Full transformer **layer** forward pass (layernorm → attn → MoE). 3 layers × 48 configs = **144** cases. |
| `../../scripts/extract_gpt_oss_matmul_activations.py` | script | Extracts per-op weight tensors from the HF checkpoint into `weight.pt` files (the matmul test's input). |
| `../../scripts/report_rel_l2.py` | script | Aggregate report — grouped mean rel_l2 tables (by op/dtype/fidelity/layer) + top movers, single-run or before/after. |
| `../../scripts/compare_rel_l2.py` | script | Row-level before/after diff of two JSONL runs, with `--min-delta` / `--filter` / `--sort-by`. |
| `../../scripts/sweeps_junit_to_rel_l2_jsonl.py` | script | Converts a tt-forge-sweeps JUnit XML (with `relative_l2`/`pcc` tags) into the JSONL format the two report scripts consume. |
| `results.md` | doc | Design rationale + **measured results** of the const-eval host-cast fix. |
| `__init__.py` | — | Package marker so `tests.operators` imports resolve. |

### Data artifacts (NOT committed)

These are produced by running the scripts/tests and are intentionally left out
of git (binary blobs / tool output). Regenerate them locally:

| Path | What | How to regenerate |
|---|---|---|
| `gpt_oss_20b_matmul_activations/` (~4.7 GB) | Per-op `input.pt` / `output.pt` / `weight.pt` + `matmul_shapes.json` from an earlier full-activation extraction | run the extraction script |
| `gpt_oss_120b_weights/` (or `/tmp/...`) | Per-op `weight.pt` files consumed by the matmul test | `extract_gpt_oss_matmul_activations.py` |
| `generated/` | tt-mlir inspector/watcher runtime output | created automatically during runs |
| `*.jsonl` | Per-test `rel_l2`/`pcc` metric logs | set `REL_L2_OUTPUT` when running a test |

---

## End-to-end workflow

```bash
cd /home/ctr-vobojevic/src/ttforge/tt-xla

# 1. Extract per-op weights once (CPU-only, ~7s per layer, needs ~16GB RAM).
python scripts/extract_gpt_oss_matmul_activations.py --output-dir /tmp/gpt_oss_120b_weights
export GPT_OSS_120B_WEIGHTS_DIR=/tmp/gpt_oss_120b_weights

# 2. Baseline run (before a compiler change) — log metrics to JSONL.
REL_L2_OUTPUT=/tmp/before.jsonl pytest tests/operators/test_matmul_gpt_oss_120b.py -s

# 3. Rebuild/install the changed wheel, then re-run.
REL_L2_OUTPUT=/tmp/after.jsonl  pytest tests/operators/test_matmul_gpt_oss_120b.py -s

# 4. Report the difference.
python3 scripts/report_rel_l2.py /tmp/before.jsonl /tmp/after.jsonl --format markdown --top 20
python3 scripts/compare_rel_l2.py /tmp/before.jsonl /tmp/after.jsonl --min-delta 0.01
```

---

## Per-component usage

### 1. Weight extraction — `extract_gpt_oss_matmul_activations.py`

Builds a temporary "fake" 1-layer checkpoint where `model.layers.{N}.*` keys are
remapped to `model.layers.0.*`, lets `from_pretrained` do the MXFP4
dequantization, then saves `q/k/v/o_proj.weight` and `mlp.router.weight` as
`layer_{N}/{op}/weight.pt`. (See `results.md` → "Why a remapped checkpoint" for
the rationale.)

```bash
# Default: layers 0 18 19 → ./gpt_oss_120b_weights
python scripts/extract_gpt_oss_matmul_activations.py

# Custom output dir / layers / explicit model dir
python scripts/extract_gpt_oss_matmul_activations.py \
    --output-dir /tmp/gpt_oss_120b_weights \
    --layers 0 18 19 \
    --model-dir ~/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/<hash>
```

### 2. Matmul test (real weights) — `test_matmul_gpt_oss_120b.py`

Requires `GPT_OSS_120B_WEIGHTS_DIR` to point at the extraction output. Each case
runs one matmul on the TT device, asserts `pcc >= 0.99`, and logs `rel_l2`.

```bash
# Smoke test (single case)
pytest tests/operators/test_matmul_gpt_oss_120b.py \
    -k "layer_0__self_attn_q_proj and opt0_bf16_hifi4_fp32true" -s -v

# By layer
pytest tests/operators/test_matmul_gpt_oss_120b.py -k "layer_0" -s -v

# All 1008 cases, logging metrics
REL_L2_OUTPUT=/tmp/run.jsonl pytest tests/operators/test_matmul_gpt_oss_120b.py -s
```

### 3. Matmul test (random weights) — `test_matmul_gpt_oss_120b_random.py`

Same shape matrix, but both operands are `torch.randn` (seed 42) — no extracted
weights needed. Use it to check whether the bfp4 error reproduces independent of
the real model weights.

```bash
REL_L2_OUTPUT=/tmp/random.jsonl pytest tests/operators/test_matmul_gpt_oss_120b_random.py -s
```

Optional env var `TT_MLIR_EXPORT_PATH` dumps per-test MLIR stages to a directory.

### 4. Layer test — `test_layer_gpt_oss_120b.py`

Loads a 1-layer model (same remap approach as extraction) and runs the full
decoder block forward on the TT device. Asserts `pcc >= 0.98` (looser than
matmul because MoE routing adds numerical error). Needs ~16GB RAM **plus** a TT
device, and reloads the model per case (~7s each).

```bash
# Smoke test
pytest tests/operators/test_layer_gpt_oss_120b.py \
    -k "layer_0 and opt0_bf16_hifi4_fp32true" -s -v

# All 144 cases
REL_L2_OUTPUT=/tmp/layer.jsonl pytest tests/operators/test_layer_gpt_oss_120b.py -s
```

### 5. Aggregate report — `report_rel_l2.py`

```bash
# Single-run summary
python3 scripts/report_rel_l2.py /tmp/run.jsonl

# Before/after, markdown, top-20 movers
python3 scripts/report_rel_l2.py /tmp/before.jsonl /tmp/after.jsonl --format markdown --top 20
```

### 6. Row-level diff — `compare_rel_l2.py`

```bash
python3 scripts/compare_rel_l2.py /tmp/before.jsonl /tmp/after.jsonl \
    --min-delta 0.01 --filter self_attn_q_proj --sort-by delta
```

### 7. Sweeps XML → JSONL — `sweeps_junit_to_rel_l2_jsonl.py`

When metrics come from a tt-forge-sweeps JUnit XML run (the `tags` property holds
`relative_l2`/`pcc`) instead of `REL_L2_OUTPUT`, convert first:

```bash
python3 scripts/sweeps_junit_to_rel_l2_jsonl.py before.xml > before.jsonl
python3 scripts/sweeps_junit_to_rel_l2_jsonl.py after.xml  > after.jsonl
python3 scripts/compare_rel_l2.py before.jsonl after.jsonl
```

---

## Compiler config matrix

All three tests share the same 48-config matrix (`optimization_level` × weight
dtype × `math_fidelity` × `fp32_dest_acc_en`). Config IDs look like
`opt0_bfp4_hifi4_fp32true`. See `results.md` → "Compiler configurations" for the
full table and the Wormhole HiFi4+fp32_acc caveat.

## JSONL metric format

One JSON object per line, appended when `REL_L2_OUTPUT` is set:

```json
{"test_id": "...::test_matmul_gpt_oss_120b[layer_0__self_attn_q_proj-opt0_bf16_hifi4_fp32true]", "rel_l2": 0.012345, "pcc": 0.998765}
```
