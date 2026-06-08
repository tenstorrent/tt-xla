### Describe the bug

- **Model / component:** HunyuanImage 2.1 (Distilled) `text_encoder_2` — the ByT5 glyph encoder (`T5EncoderModel`, ~0.22B params). Component test `tests/torch/models/HunyuanImage_2_1/test_text_encoder_2.py::test_text_encoder_2`.
- **Failure:** whole-component PCC `0.9815547776590103` vs required `0.99` — `~0.0084` under the bar. The graph **compiles and executes cleanly** (150s, no OOM, no fatal); the only failure is the final PCC assertion in the tt-xla comparison evaluator.
- **Arch / dtype:** single-device Blackhole **p150**, `fp32`, `optimization_level=1`. No sharding — this is a pure replicated single-device run, so the gap is **not** a tensor-parallel / SPMD issue.

This is the investigated version of the bare tracker **#4784** (same component, PCC `0.9827`, empty body). See § Related issues — the developer should decide whether to update #4784 in place rather than file a separate issue.

### Call chain

```
T5EncoderModel  (ByT5 glyph encoder = text_encoder_2)
  → T5Stack  (encoder blocks)
      → T5LayerSelfAttention
          → T5Attention            # NOTE: T5 omits the 1/sqrt(d) query scaling
              → matmul(Q, Kᵀ) + relative_position_bias → softmax → matmul(·, V)
                  → StableHLO matmul / softmax (fp32)
                      → ttnn matmul / softmax   # device fp32 accumulation = suspected numeric drift
```

### Key observations

- **IN:** `input_ids` `(1, 128)` int64, `attention_mask` `(1, 128)` float32. **OUT:** `last_hidden_state` `(1, 128, 1472)` float.
- PCC observed **0.9816**, required **0.99**, gap **~0.0084**. Borderline — consistent with accumulated per-op fp32 numeric drift across the encoder stack, not a single catastrophically-wrong op.
- **No OOM, no compile error** — weights (0.22B fp32) fit comfortably on p150; the run is fast (150s) and otherwise healthy.
- T5/ByT5 attention is numerically sensitive: unlike most transformers, **T5 does not scale queries by `1/sqrt(d)`** and folds scale into the learned weights, so the QKᵀ logits have a large dynamic range. Device fp32 matmul/softmax accumulation differences are amplified here, which is the leading hypothesis for the drift. No obvious loader-side fix.
- The sibling component `text_encoder` (Qwen2.5-VL) shows the **same class** of single-device device-numeric PCC gap (`0.9527`, also no OOM) — single-device does not rescue PCC there either, reinforcing that these are op-level device-numeric gaps rather than sharding bugs.

### Experiments / sanities

| Test | Result | Notes |
|------|--------|-------|
| `text_encoder_2` whole component, p150 single-device fp32 | **PCC 0.9816** | `logs/te2_iter_1_p150.log`; compiles+executes, 150s |
| Prior report (#4784) | PCC 0.9827 | same component, slightly higher; bare tracker, no investigation |
| Isolated T5 attention op sanity | _not yet run_ | needs runtime-failure-debugger op-level bisect |

### Steps to reproduce

```bash
git checkout akannan/hunyuan_image_e2e_pipeline
# single-device Blackhole p150 host, TT_VISIBLE_DEVICES=0
pytest -svv "tests/torch/models/HunyuanImage_2_1/test_text_encoder_2.py::test_text_encoder_2"
```

Failing assertion:

```
tests/infra/evaluators/evaluator.py:72: in _assert_on_results
    assert False, "\n".join(error_messages)
E   AssertionError: Evaluation result 0 failed: PCC comparison failed.
E   Calculated: pcc=0.9815547776590103. Required: pcc=0.99.
```

### Logs

- `/proj_sw/user_dev/ctr-akannan/3_jun_yyz/tt-xla/.claude/bringup/hunyuan_image_2_1/logs/te2_iter_1_p150.log`
- Decisive line (31 / 63): `AssertionError: ... pcc=0.9815547776590103. Required: pcc=0.99.`

```
tests/torch/models/HunyuanImage_2_1/test_text_encoder_2.py::test_text_encoder_2
...
2026-06-06 18:05:11 W torch_xla/csrc/runtime/profiler.cpp:88] Profiler API not found for PJRT plugin
FAILED
E   AssertionError: Evaluation result 0 failed: PCC comparison failed.
E   Calculated: pcc=0.9815547776590103. Required: pcc=0.99.
================== 1 failed, 20 warnings in 150.49s (0:02:30) ==================
```

### Expected behavior

`text_encoder_2` should produce `last_hidden_state` matching the CPU golden at `pcc >= 0.99`, the same threshold the other passing pipeline components meet, so the ByT5 glyph branch can feed the HunyuanImage transformer without degrading downstream image fidelity.

### Suggested next steps

1. **runtime-failure-debugger op-level PCC bisect** — walk the T5 encoder block-by-block / op-by-op to localize where PCC degrades (suspected: `matmul(Q, Kᵀ)` + `softmax` in `T5Attention`, given T5's unscaled-query large dynamic range).
2. **Candidate fix: force fp32 accumulation** on the attention matmul/softmax path (device math-fidelity / accumulation config) and re-measure PCC; verify the drift is accumulation-precision driven.
3. **Decision per component:** ship at measured PCC with an `xfail` (documenting the ~0.0084 device-numeric gap) vs. invest in closing the gap. Coordinate with the same decision pending for the sibling `text_encoder` component.

### Related issues

- **#4784** — `[HunyuanImage-2.1-Distilled-Diffusers] PCC drop in encoder 2` (PCC ~0.9827). Same component and failure mode as this run (0.9816). Prefer **expanding #4784** with this investigation rather than filing a duplicate.
- **#4773** — `Model Bringup - HunyuanImage-2.1` (umbrella bringup tracker; link as parent).
- Sibling component issues from the same bringup: **#4779** (encoder 1 OOM), **#4780** (transformer OOM), **#4781** (decoder OOM). The `text_encoder` (Qwen2.5-VL) PCC `0.9527` device-numeric gap is the closest analog and is **not yet filed** — same op-level numeric class, may warrant its own draft.

### Notes

- Arch: Blackhole p150, single-device, fp32, opt_level=1. Classification: **model-surface PCC**, op-level device-numeric root cause (T5 attention accumulation), not a sharding/SPMD bug.
- tt-xla issues typically need **Type: Bug** set in the GitHub UI (gh CLI cannot set it).
- Branch with repro: `akannan/hunyuan_image_e2e_pipeline`.
