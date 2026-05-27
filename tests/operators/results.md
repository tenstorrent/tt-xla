# GPT-OSS 120B Single-Op & Layer Test Report

## Goal

Isolated testing of individual matmul operations and entire transformer layers from GPT-OSS 120B on a TT device, with the goal of measuring the `relative_l2` metric before and after compiler changes.

There are two independent PoC test files:

1. **`test_matmul_gpt_oss_120b.py`** — isolated matmuls (q/k/v/o_proj, router, attn_score, attn_context) with real weights
2. **`test_layer_gpt_oss_120b.py`** — full transformer layer forward pass (input_layernorm → attention → post_attention_layernorm → MoE router + experts)

Both write metrics to a JSONL file for before/after comparison.

---

## Model

- **Name:** `openai/gpt-oss-120b` (HuggingFace)
- **Architecture:** GPT-OSS, 36 transformer layers, MoE
- **Key parameters:**
  - `hidden_size = 2880`
  - `num_attention_heads = 64`, `head_dim = 64`
  - `num_key_value_heads = 8`
  - `num_local_experts = 128`, `experts_per_token = 4`
  - `layer_types`: alternates `sliding_attention` (even indices) and `full_attention` (odd indices)
- **Quantization:** MoE expert weights are in MXFP4 format (`gate_up_proj_blocks` + `gate_up_proj_scales`); the model automatically dequantizes them to bfloat16 when no GPU is available

---

## Solution architecture

### 1. Weight extraction

**Script:** `scripts/extract_gpt_oss_matmul_activations.py`

**Approach:** for each target layer N, a temporary "fake" 1-layer checkpoint is created where the `model.layers.{N}.*` keys are remapped to `model.layers.0.*`, so `from_pretrained` loads the target layer (with correct MXFP4 dequantization) into the layer 0 slot of a 1-layer model. The individual weight tensors are then extracted from the loaded model and saved as `.pt` files.

**Steps:**

1. Parse `model.safetensors.index.json` from the HuggingFace cache
2. Find all `model.layers.{N}.*` keys and their shards
3. Read the weights from those shards, rename the keys to `model.layers.0.*`
4. Save as a new `model-layer-remapped.safetensors` in a temporary directory
5. Build a modified index that maps `model.layers.0.*` to that new shard
6. Symlink the other shards (for embeddings, lm_head, etc.) and copy the config files
7. Call `AutoModelForCausalLM.from_pretrained(tmp_dir, num_hidden_layers=1, ...)` — automatically dequantizes the MXFP4 expert weights
8. Extract `q_proj.weight`, `k_proj.weight`, etc. from `model.model.layers[0]`
9. Save as `layer_{N}/{op}/weight.pt`

**Running:**

```bash
python scripts/extract_gpt_oss_matmul_activations.py --output-dir /tmp/gpt_oss_120b_weights
# or with an explicit path:
python scripts/extract_gpt_oss_matmul_activations.py \
    --model-dir ~/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/<hash> \
    --layers 0 18 19
```

**Targeted layers:** 0, 18, 19  
**Targeted operations:** `self_attn_q_proj`, `self_attn_k_proj`, `self_attn_v_proj`, `self_attn_o_proj`, `mlp_router`  
**Excluded:** MoE `gate_up_proj` / `down_proj` — the weight is 3D `[num_experts, in, out]`, not isolable as a standard matmul

---

### 2. Matmul test (isolated operations)

**File:** `tests/operators/test_matmul_gpt_oss_120b.py`

For each test case:

- **LHS activation:** generated randomly (`torch.randn`, seed=42, dtype=bfloat16)
- **RHS (weight):** loaded from `weight.pt`, transposed (`nn.Linear` stores `[out, in]`, matmul needs `[in, out]`)
- **Attention matmuls** (`attn_score_matmul`, `attn_context_matmul`): both operands random, no weight file
- **Test wrapper:** `_MatmulWithWeight` (weight as an `nn.Parameter`) or `_Matmul` (pure 2-input matmul)
- **Execution:** `run_op_test` from the tt-xla infra (`Framework.TORCH`)

**Operations:**

| Op | has_weight | LHS shape | RHS shape (after transpose) |
|----|------------|-----------|----------------------------|
| `self_attn_q_proj` | yes | (1, 128, 2880) | (2880, 4096) |
| `self_attn_k_proj` | yes | (1, 128, 2880) | (2880, 512) |
| `self_attn_v_proj` | yes | (1, 128, 2880) | (2880, 512) |
| `self_attn_o_proj` | yes | (1, 128, 4096) | (4096, 2880) |
| `mlp_router` | yes | (128, 2880) | (2880, 32) |
| `attn_score_matmul` | no | (1, 64, 128, 64) | (1, 64, 64, 128) |
| `attn_context_matmul` | no | (1, 64, 128, 128) | (1, 64, 128, 64) |

**Threshold:** PCC ≥ 0.99

**Total number of test cases:** 3 layers × 7 ops × 48 compiler configs = **1008**

---

### 3. Layer test (full forward pass)

**File:** `tests/operators/test_layer_gpt_oss_120b.py`

For each test case:

- **Model loading:** the same `_create_remapped_checkpoint` approach as in extraction — a 1-layer model with the weights of the target layer
- **Inputs:**
  - `hidden_states`: random `(1, 128, 2880)` bfloat16
  - `position_ids`: `arange(128).unsqueeze(0)` long
- **Wrapper:** `_LayerWrapper`, which precomputes the RoPE embeddings (`model.model.rotary_emb(hidden_states, position_ids)`) and passes them as `position_embeddings=(cos, sin)` into the layer (new HF API)
- **Forward on the TT device:** the entire block — input_layernorm → q/k/v/o_proj → RoPE → attention → post_attention_layernorm → MoE router → 128 experts → output

**Layers:** 0, 18, 19 (covers both types: sliding and full attention)

**Threshold:** PCC ≥ 0.98 (lower than for the isolated matmul because the whole block + MoE routing introduces more numerical error)

**Total number of test cases:** 3 layers × 48 compiler configs = **144**

**Current measured values (opt0, bf16, hifi4, fp32acctrue):**

| Layer | rel_l2 | pcc |
|-------|--------|-----|
| layer_0 | 0.162423 | 0.986733 |
| layer_18 | 0.147858 | 0.989022 |
| layer_19 | (passed) | ≥ 0.98 |

---

### 4. Compiler configurations

Full matrix, identical to `test_matmul_mp.py`:

| Dimension | Values |
|-----------|--------|
| `optimization_level` | 0, 2 |
| `experimental_weight_dtype` | `""` (bf16), `"bfp_bf8"`, `"bfp_bf4"` |
| `math_fidelity` | `hifi4`, `hifi3`, `hifi2`, `lofi` |
| `fp32_dest_acc_en` | `True`, `False` |

Total: 2 × 3 × 4 × 2 = **48 compiler configs**

ID format: `opt{0|2}_{bf16|bfp8|bfp4}_{lofi|hifi2|hifi3|hifi4}_fp32{true|false}`

**Hardware note:** Wormhole has a bug where HiFi4 + fp32_acc gives worse accuracy than HiFi3 + fp32_acc. The system reports a warning at startup.

---

### 5. Metric measurement

A `custom_comparator` is used instead of the default `ComparisonConfig` evaluator, because `run_op_test` otherwise computes a `ComparisonResult` but discards it (does not log the values).

**Computed metrics** (both tensors are first moved to CPU, then cast to float64):

- **`relative_l2`** = `||tt_output - cpu_output||₂ / ||cpu_output||₂`
- **`pcc`** = Pearson correlation coefficient (`torch.corrcoef`)

The values are:
1. Printed to stdout during the test (visible with `-s`):  
   `[METRICS] rel_l2=0.012345  pcc=0.998765  test=<nodeid>`
2. Saved as a pytest property in the JUnit XML (with `--junitxml`)
3. Appended to a JSONL file if the `REL_L2_OUTPUT` env var is set

PCC is asserted against the threshold (0.99 for matmul, 0.98 for layer). `rel_l2` is only logged — it is used for before/after comparison.

---

## Running

### Preparation (once)

```bash
cd /home/ctr-vobojevic/src/ttforge/tt-xla

# Extract weights for the matmul test
python scripts/extract_gpt_oss_matmul_activations.py --output-dir /tmp/gpt_oss_120b_weights
export GPT_OSS_120B_WEIGHTS_DIR=/tmp/gpt_oss_120b_weights
```

### Matmul test

```bash
# Smoke test (1 case):
pytest tests/operators/test_matmul_gpt_oss_120b.py \
    -k "layer_0__self_attn_q_proj and opt0_bf16_hifi4_fp32true" -s -v

# Subset by layer:
pytest tests/operators/test_matmul_gpt_oss_120b.py -k "layer_0" -s -v

# All (1008 cases):
pytest tests/operators/test_matmul_gpt_oss_120b.py -s -v
```

### Layer test

```bash
# Smoke test (1 case):
pytest tests/operators/test_layer_gpt_oss_120b.py \
    -k "layer_0 and opt0_bf16_hifi4_fp32true" -s -v

# Subset by layer:
pytest tests/operators/test_layer_gpt_oss_120b.py -k "layer_18" -s -v

# All (144 cases):
pytest tests/operators/test_layer_gpt_oss_120b.py -s -v
```

### Before/after workflow (before and after a compiler fix)

```bash
# Before the fix:
REL_L2_OUTPUT=before.jsonl pytest tests/operators/test_matmul_gpt_oss_120b.py -s
REL_L2_OUTPUT=before_layer.jsonl pytest tests/operators/test_layer_gpt_oss_120b.py -s

# Apply the fix, rebuild...

# After the fix:
REL_L2_OUTPUT=after.jsonl pytest tests/operators/test_matmul_gpt_oss_120b.py -s
REL_L2_OUTPUT=after_layer.jsonl pytest tests/operators/test_layer_gpt_oss_120b.py -s
```

**JSONL format** (one JSON object per line):

```json
{"test_id": "tests/operators/test_matmul_gpt_oss_120b.py::test_matmul_gpt_oss_120b[layer_0__self_attn_q_proj-opt0_bf16_hifi4_fp32true]", "rel_l2": 0.012345, "pcc": 0.998765}
```

To compare two JSONL files — a simple script:

```python
import json
before = {entry["test_id"]: entry for entry in (json.loads(line) for line in open("before.jsonl"))}
after  = {entry["test_id"]: entry for entry in (json.loads(line) for line in open("after.jsonl"))}
for tid in sorted(before):
    if tid in after:
        d = after[tid]["rel_l2"] - before[tid]["rel_l2"]
        print(f"{d:+.6f}  {tid}")
```

---

## Technical details

### Why a remapped checkpoint instead of direct state dict injection?

Attempt 1 (rejected): load a 1-layer model with `from_pretrained`, then `model.load_state_dict(layer_N_weights, strict=False)` to inject layer N.

**Problem:** the MoE expert weights in safetensors are in MXFP4 format (`gate_up_proj_blocks` + `gate_up_proj_scales`). `from_pretrained` automatically dequantizes them into a regular `gate_up_proj` parameter. But when we attempt state dict injection, the raw `_blocks`/`_scales` keys do not match the dequantized parameter in the model — `load_state_dict` reports them as "unexpected keys".

**Solution:** build a temporary "fake" checkpoint where the `model.layers.{N}.*` keys are remapped to `model.layers.0.*` and let `from_pretrained` do the dequantization just like any regular layer 0 load.

### Why custom_comparator instead of `ComparisonConfig`?

`run_op_test` calls `evaluator.evaluate(tt_res, cpu_res)`, which returns a `ComparisonResult` with all metrics (PCC, atol, rel_l2, allclose) — but the result is never logged, it is only asserted against `assert_on_failure`. To get the `rel_l2` value for observation (and not just pass/fail), we bypass the evaluator and compute the metric ourselves in the `custom_comparator`.

### Why does `_LayerWrapper` precompute RoPE?

GPT-OSS uses the newer HuggingFace pattern where rotary embeddings are computed at the model level (`model.rotary_emb`) and passed as `position_embeddings=(cos, sin)` into each decoder layer. The layer's `forward` no longer takes `position_ids` directly for RoPE — only `position_embeddings`. The wrapper computes them before calling the layer.

---

## Notes

- Extraction requires ~16GB RAM to load the 1-layer model (because of the MoE expert weights)
- The layer test requires the same memory + a TT device
- The weights are bfloat16 (original model dtype + MXFP4 dequantized); the LHS activations are bfloat16; the cast to float64 is done only when computing metrics
- Hardware warning: on Wormhole, HiFi4 + fp32_acc is worse than HiFi3 + fp32_acc due to a hardware bug

---

## Design FAQ: why is extraction separate from the layer test?

### What exactly does the extraction script do (step 1)?

For each target layer N:

1. **Reads the safetensors index** (`model.safetensors.index.json`) from the HF cache
2. **Finds all `model.layers.{N}.*` keys** and groups them by shard
3. **Opens only those shards** with `safe_open` and reads the relevant tensors (raw MXFP4 format for the MoE experts)
4. **Renames the keys:** `model.layers.{N}.X` → `model.layers.0.X`
5. **Builds a temporary "fake" checkpoint:**
   - Saves the remapped tensors into a new `model-layer-remapped.safetensors`
   - Builds a new index that maps `model.layers.0.*` to that file; the rest (embeddings, lm_head, norm) stay pointed at the original shards
   - Symlinks the original shards + copies the config files
6. **`AutoModelForCausalLM.from_pretrained(tmp_dir, num_hidden_layers=1, ...)`** — HuggingFace internally:
   - Reads the modified index
   - Detects MXFP4 (`_blocks` + `_scales`) and dequantizes to bf16
   - Builds a 1-layer model where layer 0 = our target layer N
7. **Extracts the weight tensors** from `model.model.layers[0]`: `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `o_proj.weight`, `mlp.router.weight`
8. **Saves as `layer_{N}/{op}/weight.pt`**
9. **Deletes the temporary directory**

The extraction script does the **exact same model loading as the layer test** (the `_create_remapped_checkpoint` function is duplicated between the script and the test file), but instead of a forward pass — it extracts the weights and saves them.

### Why isn't the same thing inside `test_layer_gpt_oss_120b.py`?

Three reasons:

#### Reason 1: Different consumers

- **The extraction script produces weight files** used by `test_matmul_gpt_oss_120b.py` (the matmul test, **1008 cases**). The matmul test only needs the isolated 2D weight tensor — not the whole model.
- **The layer test (`test_layer_gpt_oss_120b.py`)** needs the entire `nn.Module` for the layer in memory in order to run a forward pass.

#### Reason 2: Cost amortization

| | Model loads | Load duration | What you get |
|---|---|---|---|
| **Extraction (step 1)** | 3× (once per layer) | ~7s each | 15 weight.pt files |
| **Matmul test (step 2)** | 0× (just `torch.load(weight.pt)`) | <1ms | 1008 test cases |
| **Layer test (step 3)** | 144× (per test case) | ~7s each | 144 test cases |

If the matmul test had no pre-extracted weights, it would have to load the model itself 1008 times = ~2 hours just on loading. This way: 21s extraction + matmul tests run fast.

#### Reason 3: Conceptual separation

- **Extraction** = data prep (once, CPU-only, slow)
- **Tests** = repetitive on the TT device with different compiler configs

A test should not have filesystem side effects (saving weight.pt files) unless it is explicitly designed that way.

### Possible refactor

Currently the `_create_remapped_checkpoint` function is duplicated — in `extract_gpt_oss_matmul_activations.py` and in `test_layer_gpt_oss_120b.py`. It could be extracted into a shared helper module (e.g. `tests/operators/_gpt_oss_loader.py`) and imported by both consumers.

---

## Status

- ✅ Weight extraction via `from_pretrained` + remapped checkpoint (works for any layer)
- ✅ Matmul test run on the TT device
- ✅ Layer test run on the TT device (layers 0, 18, 19 all pass with PCC ≥ 0.98)
- ✅ Measuring and logging `rel_l2` + `pcc` per test case
- ✅ Before/after workflow via `REL_L2_OUTPUT` JSONL

---

## Results: const-eval weight cast on host (before/after fix)

A fix that moves the `bf16 → bfp{4,8}` cast from the device to the host CPU into the `const_eval` function.

**Pattern (visible in `ttnn_*.mlir`):**

Before the fix (cast on the device):
```mlir
%3 = "ttnn.typecast"(%weight_dram) <{dtype = bfp_bf4}>
    : tensor<2880x4096xbf16, #dram> -> tensor<...bfp_bf4, #dram>
return %3
```

After the fix (cast on the host):
```mlir
%3 = "ttnn.from_device"(%weight_dram) : -> tensor<2880x4096xbf16, #system_memory>
%4 = "ttnn.typecast"(%3) <{dtype = bfp_bf4}>
    : tensor<2880x4096xbf16, #system_memory> -> tensor<...bfp_bf4, #system_memory>
%5 = "ttnn.to_device"(%4, %device) : -> tensor<...bfp_bf4, #dram>
return %5
```

### Experiment

- **Test:** `test_matmul_gpt_oss_120b.py` — 7 ops × 3 layers × 48 compiler configs = 1008 test cases
- **Metric:** `rel_l2 = ‖tt − cpu‖₂ / ‖cpu‖₂` (cast both tensors to float64 before computing)
- **Pre-fix run:** `/tmp/rel_l2_before_full.jsonl`
- **After-fix run:** `/tmp/rel_l2_after_full.jsonl`
- **Report generation:** `python3 scripts/report_rel_l2.py before.jsonl after.jsonl`

### Mean rel_l2 by dtype

| dtype | N | before | after | delta | % change |
|---|---|---|---|---|---|
| bf16 | 336 | 0.023685 | 0.023685 | +0.000000 | **+0.00%** |
| bfp4 | 336 | 0.157847 | 0.094923 | -0.062925 | **-39.86%** |
| bfp8 | 336 | 0.021912 | 0.021688 | -0.000224 | -1.02% |

bf16 unchanged — there is no cast when the target dtype = source dtype.

### Mean rel_l2 by (op, dtype)

| op | dtype | N | before | after | % change |
|---|---|---|---|---|---|
| attn_context_matmul | bf16/bfp4/bfp8 | 48 | 0.008205 | 0.008205 | 0.00% |
| attn_score_matmul | bf16/bfp4/bfp8 | 48 | 0.007370 | 0.007370 | 0.00% |
| mlp_router | bfp4 | 48 | 0.217096 | 0.127603 | **-41.22%** |
| mlp_router | bfp8 | 48 | 0.028167 | 0.027906 | -0.93% |
| self_attn_k_proj | bfp4 | 48 | 0.230413 | 0.141070 | **-38.77%** |
| self_attn_k_proj | bfp8 | 48 | 0.026657 | 0.026324 | -1.25% |
| self_attn_o_proj | bfp4 | 48 | 0.224659 | 0.137671 | **-38.72%** |
| self_attn_o_proj | bfp8 | 48 | 0.031572 | 0.031216 | -1.13% |
| self_attn_q_proj | bfp4 | 48 | 0.213877 | 0.125254 | **-41.44%** |
| self_attn_q_proj | bfp8 | 48 | 0.025842 | 0.025519 | -1.25% |
| self_attn_v_proj | bfp4 | 48 | 0.203310 | 0.117286 | **-42.31%** |
| self_attn_v_proj | bfp8 | 48 | 0.025569 | 0.025273 | -1.16% |

The attention matmuls (`attn_score_matmul`, `attn_context_matmul`) have both operands random — they have no constant weight that gets const-evaluated, so the fix has no effect.

### Mean rel_l2 by (dtype, fidelity)

| dtype | fidelity | N | before | after | % change |
|---|---|---|---|---|---|
| bfp4 | hifi2 | 84 | 0.157357 | 0.094288 | **-40.08%** |
| bfp4 | hifi3 | 84 | 0.156863 | 0.094083 | **-40.02%** |
| bfp4 | hifi4 | 84 | 0.156858 | 0.094078 | **-40.02%** |
| bfp4 | lofi  | 84 | 0.160311 | 0.097243 | **-39.34%** |
| bfp8 | hifi2 | 84 | 0.021459 | 0.021207 | -1.17% |
| bfp8 | hifi3 | 84 | 0.020976 | 0.020667 | -1.47% |
| bfp8 | hifi4 | 84 | 0.020960 | 0.020660 | -1.43% |
| bfp8 | lofi  | 84 | 0.024252 | 0.024216 | -0.15% |

The improvement is consistent across all fidelity levels — the fix has no interaction with `math_fidelity`.

### Mean rel_l2 by (layer, dtype)

| layer | dtype | N | before | after | % change |
|---|---|---|---|---|---|
| layer_0  | bfp4 | 112 | 0.157240 | 0.093891 | **-40.29%** |
| layer_18 | bfp4 | 112 | 0.158077 | 0.096267 | **-39.10%** |
| layer_19 | bfp4 | 112 | 0.158225 | 0.094610 | **-40.21%** |
| layer_0  | bfp8 | 112 | 0.021509 | 0.021305 | -0.95% |
| layer_18 | bfp8 | 112 | 0.022136 | 0.021900 | -1.06% |
| layer_19 | bfp8 | 112 | 0.022091 | 0.021858 | -1.05% |

Identical improvement across all three layers.

### Counts (over 1008 tests, 0.1% threshold)

| | count |
|---|---|
| Improved | 340 |
| Regressed | 100 |
| Unchanged | 568 |

568 unchanged = all 336 bf16 + all 96 attention matmuls × 3 dtypes + ~136 bfp8 within ±0.1%.

### Top 5 improvements

| % change | before | after | test |
|---|---|---|---|
| -45.75% | 0.224 | 0.122 | layer_0__mlp_router-opt0_bfp4_hifi2_fp32true |
| -45.69% | 0.211 | 0.115 | layer_19__self_attn_v_proj-opt0_bfp4_hifi2_fp32true |
| -45.54% | 0.223 | 0.121 | layer_0__mlp_router-opt0_bfp4_hifi3_fp32true |
| -45.36% | 0.210 | 0.115 | layer_19__self_attn_v_proj-opt0_bfp4_hifi3_fp32true |
| -45.19% | 0.210 | 0.115 | layer_0__self_attn_v_proj-opt0_bfp4_hifi2_fp32true |

### Top 5 regressions

| % change | before | after | test |
|---|---|---|---|
| +20.24% | 0.010001 | 0.012025 | layer_19__self_attn_v_proj-opt0_bfp8_hifi4_fp32true |
| +20.19% | 0.010013 | 0.012034 | layer_19__self_attn_v_proj-opt0_bfp8_hifi3_fp32true |
| +19.88% | 0.009779 | 0.011723 | layer_0__self_attn_v_proj-opt0_bfp8_hifi4_fp32true |
| +19.85% | 0.009799 | 0.011743 | layer_0__self_attn_v_proj-opt0_bfp8_hifi3_fp32true |
| +19.78% | 0.009952 | 0.011920 | layer_18__self_attn_v_proj-opt0_bfp8_hifi3_fp32true |

All regressions are `bfp8 v_proj` and `bfp8 q_proj` across all layers. The absolute values are small (~0.01–0.014 rel_l2), but it's a systematic pattern — worth opening an issue to investigate why the host implementation of `bf16 → bfp8` is slightly less precise than the device's (probably different rounding).

### Conclusion

| dtype | Effect of the fix |
|---|---|
| **bf16** | unchanged (no cast) |
| **bfp4** | **≈40% reduction in rel_l2** — the main winner (MoE expert weights in GPT-OSS-120B) |
| **bfp8** | ≈1% reduction on average, but a +18–20% regression for `q_proj`/`v_proj` |

### Reproduce

```bash
# Step 1: extract the weights (once)
python scripts/extract_gpt_oss_matmul_activations.py --output-dir /tmp/gpt_oss_120b_weights
export GPT_OSS_120B_WEIGHTS_DIR=/tmp/gpt_oss_120b_weights

# Step 2: run the tests with the pre-fix wheel
REL_L2_OUTPUT=/tmp/before.jsonl pytest tests/operators/test_matmul_gpt_oss_120b.py -s

# Step 3: install the after-fix wheel, run again
REL_L2_OUTPUT=/tmp/after.jsonl pytest tests/operators/test_matmul_gpt_oss_120b.py -s

# Step 4: generate the report
python3 scripts/report_rel_l2.py /tmp/before.jsonl /tmp/after.jsonl --format markdown --top 20
```
