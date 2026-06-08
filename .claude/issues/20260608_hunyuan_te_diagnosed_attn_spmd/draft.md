### Describe the bug

- Model: **HunyuanImage 2.1 `text_encoder`** = Qwen2.5-VL-7B `language_model` (`Qwen2_5_VLTextModel`, `model_type=qwen2_5_vl_text`, 7.07B). Test node `tests/torch/models/HunyuanImage_2_1/test_text_encoder.py::test_text_encoder_sharded`.
- Arch: `n300-llmbox` (Wormhole), mesh `(1, 4)`, fp32, ~7.07 GiB/chip. **Compiles and executes end-to-end — no OOM.** The failure is numeric (PCC), not memory or compile.
- Observed: full 4-way Megatron TP gives **PCC = 0.2774** vs required **0.99**. Replacing attention sharding with replication (MLP-only TP) recovers to **0.9637**; pure single-device (no sharding) lands at **0.9527**.
- **Two distinct effects** (see § Experiments):
  1. **Catastrophic SPMD sharding bug** — sharding the attention q/k/v across the head dim collapses PCC to 0.277. This is a sharding-propagation correctness bug, not numerics.
  2. **Residual device numeric gap** — even fully replicated attention only reaches ~0.95–0.96, so fixing the sharding bug alone will **not** reach 0.99. There is a separate op-level fp32 numeric gap in this attention path.

### Call chain

```
test_text_encoder_sharded                              # mesh (1,4), Megatron spec
  → Qwen2_5_VLTextModel.forward
      → Qwen2_5_VLAttention (GQA: 28 q heads, 4 kv heads, head_dim 128)
          → repeat_kv  (modeling_qwen2_5_vl.py:170)
                expand+reshape [bs,4,seq,d] → [bs,28,seq,d] on the SPMD-sharded kv-head dim
                ↳ sharding NOT tracked through expand/merge  ← corruption #1
          → apply_multimodal_rotary_pos_emb  (modeling_qwen2_5_vl.py:627)
                M-RoPE split/cat on head_dim, mrope_section [16,24,24]
                cos/sin broadcast over the sharded head dim  ← corruption #2
      → down_proj cross-shard all-reduce sum (fp32)        ← residual numeric gap
```

### Key observations

- **Shard spec is byte-for-byte identical** to the known-good `llama/causal_lm` and `qwen_3/causal_lm` Megatron specs (`q/k/v/gate/up=(model,batch)`, `o/down=(batch,model)`, norms `(batch,)`). Divisibility is clean at 4-way: 28 q heads /4 = 7, 4 kv heads /4 = 1, hidden 3584/4 = 896, intermediate 18944/4 = 4736; GQA groups align contiguously (chip *i*: q heads 7*i*..7*i*+6 + kv head *i*). **The partition_spec is correct.**
- **Why this model and not llama/qwen3:** those references use standard RoPE (`rope_scaling=None`) and pass with the identical spec. Qwen2.5-VL adds **M-RoPE** (`mrope_section [16,24,24]`) + GQA `repeat_kv` — the qwen2_5_vl-specific attention path is what breaks under sharding.
- **MLP-only isolates the cause:** replicating q/k/v/o and sharding only MLP gate/up/down recovers PCC 0.277 → 0.964. So MLP sharding is correct; **attention head-sharding is the dominant corruption.**
- **Single-device refutes "sharding is the whole story":** a later replicated, no-sharding run on p150 (Blackhole, fp32, fits) scored **0.9527 — worse than MLP-only TP (0.9637)**. So even with no attention sharding at all, the model does not reach 0.99. The ~0.04–0.05 residual is a **device op-level fp32 numeric gap** in M-RoPE/GQA attention, independent of SPMD.

### Experiments / sanities

| Config | Sharding | PCC | Interpretation |
|--------|----------|-----|----------------|
| Full Megatron TP `(1,4)` | q/k/v/o + MLP sharded | **0.2774** | catastrophic — SPMD mispropagation through head-sharded `repeat_kv` + M-RoPE |
| MLP-only TP `(1,4)` (`HUNYUAN_TE_MLP_ONLY=1`) | attention replicated, MLP sharded | **0.9637** | attention sharding is the dominant corruption; MLP spec correct |
| Single-device (p150) | none (replicated) | **0.9527** | device numeric gap persists with zero sharding → separate op-level issue |
| Required | — | **0.99** | — |

### Steps to reproduce

```bash
# tt-xla, branch akannan/hunyuan_image_e2e_pipeline
export TT_XLA_ARCH=n300-llmbox
export TT_VISIBLE_DEVICES=0,1,2,3
export TT_XLA_SPMD=1
export CONVERT_SHLO_TO_SHARDY=1

# (1) Catastrophic full-Megatron TP — PCC 0.277
pytest -svv "tests/torch/models/HunyuanImage_2_1/test_text_encoder.py::test_text_encoder_sharded"

# (2) MLP-only diagnostic (attention replicated) — PCC 0.964
HUNYUAN_TE_MLP_ONLY=1 pytest -svv "tests/torch/models/HunyuanImage_2_1/test_text_encoder.py::test_text_encoder_sharded"

# (3) Single-device replicated (p150, Blackhole) — PCC 0.9527
TT_VISIBLE_DEVICES=0 pytest -svv "tests/torch/models/HunyuanImage_2_1/test_text_encoder.py::test_text_encoder"
```

### Logs

- Full Megatron (0.277): `/proj_sw/user_dev/ctr-akannan/3_jun_yyz/tt-xla/.claude/bringup/hunyuan_image_2_1/logs/text_encoder_iter_1_first_run_tp.log`
- MLP-only (0.964): `/proj_sw/user_dev/ctr-akannan/3_jun_yyz/tt-xla/.claude/bringup/hunyuan_image_2_1/logs/text_encoder_iter_2_mlp_only.log`
- Single-device (0.9527): `/proj_sw/user_dev/ctr-akannan/3_jun_yyz/tt-xla/.claude/bringup/hunyuan_image_2_1/logs/te_iter_1_p150.log`
- Diagnosis: `/proj_sw/user_dev/ctr-akannan/3_jun_yyz/tt-xla/.claude/bringup/hunyuan_image_2_1/diagnosis_text_encoder.json`

Decisive excerpt (full Megatron):

```
tests/infra/evaluators/evaluator.py:72: in _assert_on_results
    assert False, "\n".join(error_messages)
E   AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.27736377316915883. Required: pcc=0.99.
```

### Expected behavior

A correct 4-way Megatron tensor-parallel partition of a GQA + M-RoPE attention stack should produce numerically equivalent results to the single-device computation (PCC → 1.0 modulo device fp32 noise). SPMD/Shardy should track sharding through the `repeat_kv` expand+reshape and the M-RoPE split/cat on `head_dim` so that head-sharded q/k tensors stay correctly partitioned. Separately, the replicated attention path should reach the 0.99 bar on device fp32.

### Suggested next steps

1. **Escalate the SPMD attention-TP bug:** file/track a torch-xla / tt-mlir SPMD issue for sharding mispropagation through head-sharded `repeat_kv` (expand+reshape, `modeling_qwen2_5_vl.py:170`) and M-RoPE reshape (`apply_multimodal_rotary_pos_emb`, `:627`) for `qwen2_5_vl`. This is what collapses PCC to 0.277 and blocks correct 4-way attention TP.
2. **Close the residual numeric gap with op-level PCC bisect** (runtime-failure-debugger): even replicated/MLP-only caps at ~0.95–0.96, so an op-level bisect on `apply_multimodal_rotary_pos_emb` + `repeat_kv` + the `down_proj` cross-shard all-reduce sum is needed to reach 0.99 regardless of the sharding fix.
3. **Interim option — MLP-only TP** (`HUNYUAN_TE_MLP_ONLY=1`): replicate q/k/v/o, shard MLP only; fits memory (~9 GiB/chip fp32). Currently 0.9637 — still under 0.99, so it only ships once step 2 closes the residual gap.
4. **Note:** "move to p150 single-device" is **not** a pass path — the 2026-06-06 single-device run scored 0.9527 (< MLP-only 0.9637), confirming the gap is op-level numerics, not sharding alone.

### Related issues

- **tenstorrent/tt-mlir#6313** — Shardy `updateShapes` / sharding mispropagation through reshape (same failure class as head-sharded `repeat_kv` + M-RoPE).
- **tenstorrent/tt-mlir#5290** — analogous Shardy mispropagation through reshape on a multichip TP path (closed).
- `#3508` — Qwen3.5 (27B+) model bringup (OPEN). Qwen-family multichip context only; standard RoPE, not M-RoPE/GQA.
- **tenstorrent/tt-mlir#3370** — HunyuanImage transformer compile blocker (separate component; unrelated to this PCC bug).
- No similar issues found in `tenstorrent/tt-xla` for qwen2_5_vl M-RoPE + GQA attention TP PCC at time of investigation.

### Notes

- Arch: `n300-llmbox` (Wormhole) for TP; p150 (Blackhole) for the single-device sanity. Classification: **model-level PCC**, two root causes — (a) SPMD attention-sharding correctness (catastrophic), (b) device op-level fp32 numerics (residual). Consider splitting into two issues if owners differ (SPMD/compiler vs op-numerics).
- A reversible, env-gated diagnostic branch `HUNYUAN_TE_MLP_ONLY` lives in `shard_text_encoder_specs` (`third_party/tt_forge_models/hunyuan_image_2_1/pytorch/loader.py`); default off = original full Megatron spec. No change to default behavior.
- tt-xla issues need **Type: Bug** set in the GitHub UI (gh CLI cannot set it).
