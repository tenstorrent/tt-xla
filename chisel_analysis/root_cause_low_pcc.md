# Root-cause analysis: why deepseek_v3_1 shows low PCC

**TL;DR:** The chisel report's low PCC is **not** evidence of a device compute bug. The
accumulated-mode failures (9,853 of them) are overwhelmingly **poisoned by the softmax golden bug**
— it corrupts the golden for every op downstream of each attention softmax. In *isolated* mode (each
op fed a correct golden input) **no arithmetic kernel ever fails**. The only genuine unknowns are two
chisel blind spots: the **bfp_bf8 weight matmuls** and the **MoE routing collectives**, both of which
chisel cannot validate because their goldens are missing/unsupported.

## Evidence

### 1. The softmax golden bug poisons the whole accumulated chain
First divergence in execution order is always the attention softmax, and the ops that consume it
inherit the corruption verbatim:

Binary 1 (prefill), device 0:
```
softmax  %200  pcc=0.258   <- chisel golden bug (dim=3 ignored, golden does dim=1)
typecast %201  pcc=0.258   <- consumes %200
matmul   %204  pcc=0.769   <- AV matmul: %204 = matmul(%201, %203), i.e. softmax_probs @ V
permute  %205  pcc=0.769   <- inherits
```
Binary 3 (decode), device 0: identical shape — `softmax %171 → typecast %172 → matmul %175 (AV)`.

IR proof (prefill `run2732_g0`):
```
%200 = ttnn.softmax(%199) {dimension = 3}      // attention probabilities
%201 = ttnn.typecast(%200) -> bf16
%204 = ttnn.matmul(%201, %203)                 // probs @ V  — attention output
```
In **accumulated** mode `%204`'s *golden* is computed from the wrong-axis golden softmax, so its 0.77
PCC is a false failure. The *device* `%204` receives the correct device softmax and is fine. Every
op after the first attention softmax in every layer (output proj, MLP, MoE, residual adds…) is
downstream of this poison — which is why the accumulated failure count explodes to ~9.8k.

**Implication:** the accumulated trace is untrustworthy until the softmax golden is fixed and the
report is re-run. The "per-layer attention matmul failures" (`%204`, `%339`, `%474`, `%609`) are all
this same artifact.

### 2. Isolated mode finds NO genuine compute-kernel bug
Isolated mode feeds each op a correct golden input, so it is immune to upstream poisoning. The
complete set of ops that fail in isolated mode:

| op | verdict |
|----|---------|
| `ttnn.softmax` | golden bug (dim attr dropped) — FALSE POSITIVE, fix known |
| `ttnn.moe_expert_token_remap` | golden dtype bug (expects uint16, got bf16) |
| `ttnn.all_to_all_dispatch` | MoE collective; golden ordering/layout suspect |
| `ttnn.to_memory_config` | layout-comparison artifact (no math; isolated≈accumulated PCC) |
| `ttnn.mean`, `ttnn.rsqrt` | metric artifacts — atol≈1e-5/0.12, values agree, PCC degenerate on 16×1 |
| `ttnn.topk` (indices) | index-PCC noise under ties |

No `matmul`, `add`, `multiply`, `sub`, `silu`, `embedding`, `rms_norm`, `concat`, … ever fails
isolated. Every genuine arithmetic kernel is correct given correct input.

### 3. The two real blind spots
chisel cannot see these, so it can neither blame nor exonerate them:
- **bfp_bf8 weight matmuls** — 428 in the prefill graph. chisel's `mlir_type_to_torch_dtype` can't
  map `!ttcore.tile<32x32, bfp_bf8>`, so all matmul/typecast/sparse_matmul numerics on the weight
  path are skipped (148 `_default_pre_op` chisel_bug records). This is exactly where DeepSeek's
  precision lives.
- **MoE token routing** — `all_to_all_dispatch` / `all_to_all_combine` (golden missing) /
  `moe_expert_token_remap` (golden dtype wrong). Device correctness here is simply unknown.

## Measured model PCC (the real symptom)
From `debug_test_deepseek_v3_1_tp_galaxy_4_layers_jn_3_v5.log`:
```
required_pcc = 0.9
Prefill PCC verification passed with PCC=0.916316
First decode PCC verification passed with PCC=0.965493
WARNING ... Device prefill produced different tokens than CPU prefill;
        using CPU prefill output as decode PCC reference.
```
PCC ≈ 0.92 (not ≈ 0) is the signature of **accumulated precision loss**, not a broken op — a
mis-routed MoE or wrong kernel would tank PCC toward 0. It passes the 0.9 bringup gate but the
prefill argmax token flips vs CPU, so generated text diverges. Prefill (seq=128, 0.916) is worse than
decode (seq=1, 0.965), consistent with error accumulating across more attention/MoE interactions.

This matches the chisel evidence exactly: every kernel is correct given correct input (isolated mode),
yet the end-to-end output drifts — i.e. the loss is distributed across many slightly-lossy ops, with
the **bfp_bf8 weight matmuls** (chisel's blind spot) the leading single contributor. Compute config is
already maxed (`math_fidelity = hifi4`, `fp32_dest_acc_en = true`), so the remaining lever is the
weight *storage* precision (bfp8), not the matmul accumulation.

## Component isolation (real model, real bf8) — attention is CLEAN, MoE is the suspect
This is a **regression** (bf8 PCC was good before), so there is a specific cause even though it shows
up under bf8. Component tests (`test_deepseek_v3_1_components.py`, real weights + real bf8 conversion,
the path that actually applies bfp8) localize it:

| component | measured PCC (bf8) | verdict |
|-----------|--------------------|---------|
| MLA attention prefill | **1.000000** | clean — bf8 attention matmuls are fine |
| MLA attention decode  | **0.999794** | clean |
| MoE block             | **not yet measured** (OOM'd at batch 64) | prime suspect |
| dense MLP             | not isolated | secondary |

**Depth-scaling argument (decisive):** `first_k_dense_replace=3`, so 4 layers = 3 dense + **1 MoE**
(→ 0.92) and 30 layers = 3 dense + **27 MoE** (→ 0.08). The dense count is fixed at 3; only the MoE
count scales with depth. So the per-layer error that compounds to 0.08 must come from the **MoE block**,
not attention or the (few, fixed) dense layers. bf8 attention being a perfect 1.0 also proves bf8 weight
quantization is not generically the problem — it's something specific to the MoE path.

**Next measurement (ready):** the MoE component test now runs at `MOE_BATCH_SIZE=8` (was OOMing at 64;
per-token numerics are batch-independent so PCC is still representative):
`pytest -svv tests/torch/models/deepseek_v3_1/test_deepseek_v3_1_components.py::test_deepseek_v3_1_moe`
→ look for `[pcc] moe: ...`. If low, drill into the MoE sub-stages (router/gate topk → all_to_all_dispatch
→ expert MLPs → all_to_all_combine); the chisel report already flagged broken goldens on dispatch/combine.

## Op-test isolation attempt (run 1 & 2): does NOT engage bfp8 — use the real model path
The standalone op/tower test (`test_deepseek_v3_1_bfp8_blindspot.py`) produced **bit-identical PCC
for bfp_bf8 and bf16 in every case**, even with `experimental_weight_dtype="bfp_bf8"` set globally.
Reason: in the single-device op-test compile path the `nn.Parameter` weights are baked as graph
**constants**, so there is no weight-tagged function argument for the tt-mlir weight-dtype conversion
pass to target (the pass annotates block args, see `shlo_input_role_propagation.cc`). bfp8 conversion
only happens on the **full model path** (`apply_weight_dtype_overrides` keeps params as inputs + the
global option). Lesson: bfp8 must be measured through the real model, not hand-built op modules. (The
run-1 bf16 control is still valid: a 30-deep bf16 tower only fell to 0.997, ruling out bf16 activation
drift as the model's failure mode.)

## What to do next (model PCC ≈ 0.92 @4 layers, ~0.08 @30 — bfp8 weights the prime suspect)
1. **Decisive experiment — run the real 4-layer benchmark with bf16 weights (APPLIED).**
   `test_deepseek_v3_1_tp_galaxy_4_layers` now passes `experimental_weight_dtype=""` (TEMP; revert
   before commit) which disables the bfp8 conversion. Re-run it and compare to the bfp_bf8 baseline
   (prefill 0.916 / decode 0.965):
   - PCC jumps to ≈0.99 → **bfp8 weight quantization is the dominant cause** (expected, lossy storage
     format). Then it's a precision/perf tradeoff decision, not a bug. Confirm by also running at
     higher layer counts (where bfp8 should now NOT collapse to 0.08).
   - PCC stays ≈0.92 → bfp8 is not it; move to per-layer PCC probing and the MoE path.
2. **Per-layer / per-block PCC probe.** Capture the hidden state after each of the 4 layers (device vs
   CPU) to see whether error grows linearly (uniform precision drift) or jumps at a specific block
   (attention vs MoE). Linear growth confirms accumulation; a jump localizes a real op.
### Isolation tests (built)
- **`tests/torch/models/deepseek_v3_1/test_deepseek_v3_1_bfp8_blindspot.py`** (new, single-device,
  no Galaxy/chisel) — directly measures the bfp8 blind spot:
  - `test_bfp8_matmul_per_projection`: one matmul per real DeepSeek projection shape, `bfp_bf8` vs
    `bf16` weight, PCC printed. Quantifies per-matmul bfp8 loss.
  - `test_bfp8_mlp_tower_depth`: residual SwiGLU tower swept over depth {1,2,4,8,16,30}, `bfp_bf8` vs
    `bf16`. Reproduces the PCC-vs-depth decay; bf16 should stay ~1.0, attributing the decay to bfp8.
  - Uses `torch.ops.tt.weight_dtype_override` (pass-through on CPU → bf16 reference, bfp8 on device),
    `math_fidelity=hifi4` + `fp32_dest_acc_en=true` to match the model. Run:
    `pytest -svv tests/torch/models/deepseek_v3_1/test_deepseek_v3_1_bfp8_blindspot.py`
  - **Run 1 finding (important):** the per-tensor override is gated on the global
    `experimental_weight_dtype` option. With it unset, bfp8 and bf16 arms were bit-identical (override
    inert). Test now sets `experimental_weight_dtype="bfp_bf8"` globally so the conversion pass runs.
    The run-1 **bf16 tower** result is still useful as a control: depth-30 PCC was **0.997**, i.e. a
    pure bf16 (no-bfp8) tower barely degrades over 30 layers — so the model's 30-layer 0.08 is **not**
    bf16 activation drift; it must be bfp8 weights and/or attention/MoE. Re-run the corrected test to
    quantify the bfp8 arm.
- **`tests/torch/models/deepseek_v3_1/test_deepseek_v3_1_components.py`** (existing, Galaxy) — already
  isolates the MLA attention (prefill/decode) and the sparse MoE block with faithful sharding.

### Next
3. **Make chisel able to see the truth (parallel track).** Fix the goldens and re-run so accumulated
   mode stops lying: (a) softmax `dimension` (1-line, `softmax_golden_fix.md`),
   (b) `moe_expert_token_remap` dtype, (c) implement `all_to_all_combine_golden`, (d) add `bfp_bf8` to
   chisel's `mlir_type_to_torch_dtype`. (d) is what would let chisel finally quantify the bfp8 loss
   directly.
