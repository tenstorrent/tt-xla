### Describe the bug

- **Model key / component:** HunyuanImage-2.1-Distilled (`hunyuanvideo-community/HunyuanImage-2.1-Distilled-Diffusers`) → `text_encoder` = Qwen2.5-VL-7B `language_model` (`Qwen2_5_VLTextModel`, ~7.07 B). Test: `tests/torch/models/HunyuanImage_2_1/test_text_encoder.py::test_text_encoder_sharded`.
- **Failure:** 4-way tensor-parallel (SPMD, mesh `(1,4)`, fp32) compiles and executes end-to-end with **no OOM**, but fails the model PCC check: **`pcc=0.27736` vs required `0.99`**. The surface error is the evaluator `AssertionError` (`tests/infra/evaluators/evaluator.py:72`); there is no `TT_FATAL` — the run is numerically wrong, not crashing.
- **Arch / config:** `n300-llmbox` (Wormhole, 4 chips via `TT_VISIBLE_DEVICES=0,1`), fp32, `CONVERT_SHLO_TO_SHARDY=1`, `optimization_level=1`. The `TT_FATAL: ... eth core connects to a remote mmio device` lines in the log are the usual non-fatal llmbox topology criticals — the run continues past them.
- **Localization:** the shard spec is **provably correct** (byte-for-byte identical to the known-good `llama/causal_lm` and `qwen_3/causal_lm` Megatron specs, divisibility OK at 4-way). The corruption is isolated to **sharded attention** — replicating attention and sharding only the MLP recovers PCC from 0.277 → 0.964 (see sanity table). The Qwen2.5-VL-specific bits absent from the passing llama/qwen3 references are **M-RoPE** (`mrope_section [16,24,24]`) and **GQA `repeat_kv`**.

### Call chain

```
Qwen2_5_VLTextModel (text_encoder, returned as .language_model)
  → Qwen2_5_VLDecoderLayer × N
      → Qwen2_5_VLAttention
          → repeat_kv(...)                          # modeling_qwen2_5_vl.py:170
          │     expand+reshape [bs,4,seq,d] -> [bs,28,seq,d] on the SPMD-sharded kv-head dim
          → apply_multimodal_rotary_pos_emb(...)     # modeling_qwen2_5_vl.py:627
          │     M-RoPE split/cat on head_dim, mrope_section [16,24,24]; cos/sin broadcast over sharded head dim
              → StableHLO reshape / concat / broadcast under Shardy (CONVERT_SHLO_TO_SHARDY=1)
                  → ttnn.mesh_shard / sharding propagation   # sharding mistracked through expand+merge reshape
```

### Key observations

- **Spec is not the bug.** At 4-way: 28 q-heads/4 = 7, 4 kv-heads/4 = 1, hidden 3584/4 = 896, intermediate 18944/4 = 4736 — all divide cleanly; GQA groups align contiguously (chip *i* gets q-heads `7i..7i+6` + kv-head *i*). The same spec passes for `llama` and `qwen_3`.
- **Attention sharding is the dominant corruption.** Replicating q/k/v/o and sharding only the MLP (`HUNYUAN_TE_MLP_ONLY=1`) lifts PCC `0.277 → 0.9637`. So MLP TP is correct; the loss is entirely in the head-sharded attention path.
- **It is *also* a device op-level numeric issue, independent of sharding.** The same model **single-device, fully replicated** on p150 (no sharding at all) lands at **PCC 0.9527** — *worse* than MLP-only TP. So there are two stacked problems: (1) a baseline device-numeric gap in the M-RoPE/GQA attention even when replicated, and (2) an additional catastrophic collapse when the attention is head-sharded under SPMD. Moving to single-device does **not** rescue PCC.
- **Suspected upstream mechanism:** torch-xla/Shardy mispropagates the sharding annotation through the two attention reshapes on the head-sharded q/k tensors — `repeat_kv`'s `expand`+`reshape` (`[bs,4,seq,d]→[bs,28,seq,d]`) and the M-RoPE `split`/`cat` on `head_dim`. This is the same *class* as tt-mlir reshape/sharding-propagation bugs (#6313, #5290).

### Experiments / sanities

| Test | PCC | Notes |
|------|-----|-------|
| Full Megatron TP, mesh `(1,4)`, fp32 | **0.2774** | this log (`text_encoder_iter_1_first_run_tp.log`) |
| MLP-only TP (`HUNYUAN_TE_MLP_ONLY=1`, attn replicated) | 0.9637 | `logs/text_encoder_iter_2_mlp_only.log` — isolates gap to sharded attention |
| Single-device replicated, p150, fp32 (no sharding) | 0.9527 | `logs/te_iter_1_p150.log` — baseline device-numeric gap; refutes "single-device rescues PCC" |

### Steps to reproduce

```bash
git checkout akannan/hunyuan_image_e2e_pipeline

# 4-way tensor-parallel repro (this issue) — needs an n300-llmbox / 4 visible chips
TT_XLA_ARCH=n300-llmbox TT_VISIBLE_DEVICES=0,1 TT_XLA_SPMD=1 CONVERT_SHLO_TO_SHARDY=1 \
  pytest -svv "tests/torch/models/HunyuanImage_2_1/test_text_encoder.py::test_text_encoder_sharded"

# Sanity A — MLP-only TP (attention replicated) -> PCC ~0.964
HUNYUAN_TE_MLP_ONLY=1 TT_XLA_ARCH=n300-llmbox TT_VISIBLE_DEVICES=0,1 TT_XLA_SPMD=1 CONVERT_SHLO_TO_SHARDY=1 \
  pytest -svv "tests/torch/models/HunyuanImage_2_1/test_text_encoder.py::test_text_encoder_sharded"

# Sanity B — single-device replicated on p150 (no sharding) -> PCC ~0.953
pytest -svv "tests/torch/models/HunyuanImage_2_1/test_text_encoder.py::test_text_encoder"
```

Failing excerpt (this log):

```
tests/infra/evaluators/comparison_evaluator.py:73: in evaluate
    ComparisonEvaluator._assert_on_results(_comparison_result)
tests/infra/evaluators/evaluator.py:72: in _assert_on_results
    assert False, "\n".join(error_messages)
E   AssertionError: Evaluation result 0 failed: PCC comparison failed.
    Calculated: pcc=0.27736377316915883. Required: pcc=0.99.
```

### Logs

- `/proj_sw/user_dev/ctr-akannan/3_jun_yyz/tt-xla/.claude/bringup/hunyuan_image_2_1/logs/text_encoder_iter_1_first_run_tp.log` — **primary**; line 50 / line 82 = `pcc=0.27736 Required: pcc=0.99`. Device init criticals (lines 23–30) are non-fatal llmbox topology warnings.
- `/proj_sw/user_dev/ctr-akannan/3_jun_yyz/tt-xla/.claude/bringup/hunyuan_image_2_1/logs/text_encoder_iter_2_mlp_only.log` — MLP-only sanity (PCC 0.9637).
- `/proj_sw/user_dev/ctr-akannan/3_jun_yyz/tt-xla/.claude/bringup/hunyuan_image_2_1/logs/te_iter_1_p150.log` — single-device replicated sanity (PCC 0.9527).
- Diagnosis JSON: `/proj_sw/user_dev/ctr-akannan/3_jun_yyz/tt-xla/.claude/bringup/hunyuan_image_2_1/diagnosis_text_encoder.json`.

### Expected behavior

4-way tensor-parallel `text_encoder` should produce `last_hidden_state` matching the CPU golden at `pcc ≥ 0.99`, the same bar `llama`/`qwen_3` meet with the identical Megatron shard spec. Head-sharded `repeat_kv` and M-RoPE reshapes must preserve sharding semantics through `expand`/`reshape`/`split`/`cat` so the sharded attention output equals the replicated result.

### Suggested next steps

1. **Escalate the SPMD/Shardy reshape-propagation bug upstream (tt-mlir).** File/link a tt-mlir issue for head-sharded `repeat_kv` (`expand`+`reshape`) and M-RoPE `split`/`cat` mispropagation on the qwen2_5_vl attention path — this blocks correct 4-way attention TP and is the root fix. Cross-link the reshape/sharding-propagation issues #6313 and #5290.
2. **Op-level PCC bisect (runtime-failure-debugger)** on `apply_multimodal_rotary_pos_emb` (`modeling_qwen2_5_vl.py:627`) and `repeat_kv` (`:170`) to pinpoint where the sharded result diverges from the replicated one, and to quantify the *replicated-only* baseline gap (the 0.9527 single-device number).
3. **Interim ship path — MLP-only TP** (`HUNYUAN_TE_MLP_ONLY=1`, ~0.964): viable only after closing the residual ~0.026 device-numeric gap (down_proj cross-shard all-reduce + replicated attention numerics) to clear 0.99. Note: single-device on p150 does **not** clear the bar (0.9527), so it is not a passing fallback.

### Related issues

- **tenstorrent/tt-mlir#6313** — `updateShapes` does not propagate expected shapes based on shardings on defining op (OPEN, bug). Same class as the suspected root cause: sharding/shape mispropagation through reshape.
- **tenstorrent/tt-mlir#5290** — Matmul bias set to replicate gets sharded in ttir, breaking a downstream `reshape` (CLOSED). Closely analogous Shardy mispropagation through reshape on a multi-chip TP path.
- **tenstorrent/tt-xla#4581** — `qwen_2_5_vl/pytorch-3B_Instruct` (OPEN). Same model family (Qwen2.5-VL); different failure (L1 CB overflow), tracked for context.
- **tenstorrent/tt-xla#4780** — `[HunyuanImage-2.1-Distilled-Diffusers] OOM in transformer` (OPEN). Companion component in the same pipeline bringup; not the same root cause (this is PCC, that is activation OOM).

### Notes

- Arch: `n300-llmbox`, 4 chips, fp32, SPMD + Shardy. Classification: **model-surface PCC on tt-xla**, with a suspected **upstream tt-mlir/Shardy** root cause for the TP-specific collapse plus an independent device-numeric gap in the replicated attention.
- A reversible, env-gated diagnostic branch (`HUNYUAN_TE_MLP_ONLY`) exists in the loader's `shard_text_encoder_specs` (default off = full Megatron spec); used only for the MLP-only sanity above.
