### Describe the bug
- `playground_v2_5/text_encoder_2` (`CLIPTextModelWithProjection`, 32 encoder blocks) — whole-model PCC = **0.9711294738181017**, required ≥ **0.99** (atol = 2.0833).
- Per-layer cumulative sweep: PCC is clean at the top of the stack (iter 0 ≈ 0.999), crosses the 0.99 threshold around block **~13**, and degrades to ~0.979 by the final block. No single layer is responsible — error compounds through the encoder.
- Random tokens reproduce the failure with the real model + real weights; arch = n150.

#### Call chain (where it shows up)
```
TextEncoder2Wrapper
  → CLIPTextModelWithProjection
      → CLIPTextTransformer
          → CLIPEncoder (32 × CLIPEncoderLayer)   # PCC drifts here, cumulatively
      → Linear(1280 → 1280, bias=False)           # text_projection
```

#### Key observations
- Whole-model only repro — no single encoder block fails in isolation; the drop is cumulative across the 32-layer stack.
- iter 0 (1 block) PCC ≈ 0.999 → still passes threshold.
- iter 13 (~14 blocks) PCC drops below 0.99 → first failing depth.
- iter 31 (full 32-block stack) PCC ≈ 0.979 → matches whole-model 0.9711.
- Input is random token ids; CPU golden vs TT device output is compared in `run_graph_test` with `assert_on_failure=True`.
- Same compounding pattern as prior CLIP / decoder-stack PCC issues (cf. tt-xla#4849, #4328 — cumulative error in a repeated transformer block).

#### Experiments / sanities
| Test | PCC | Notes |
|------|------|-------|
| whole model (32 blocks) | 0.9711 | below 0.99 threshold |
| cumulative iter 31 (full stack) | ~0.979 | matches whole model |
| cumulative iter 13 (~14 blocks) | < 0.99 | first failing depth |
| cumulative iter 0 (1 block) | ~0.999 | passes threshold |

Per-block layer-depth sweep (cumulative PCC vs blocks executed on device, remainder on CPU) confirms the failure is a slow drift through the encoder, not a single bad op.

### Steps to reproduce the issue
```bash
git checkout kkannan/may21_playground_v2_5_encoder_2_pcc_drop
pytest -svv tests/torch/models/playground_v2_5/test_text_encoder_2.py::test_text_encoder_2
```

Failing traceback excerpt:
```
tests/infra/evaluators/evaluator.py:72: AssertionError
E   AssertionError: Evaluation result 0 failed: PCC comparison failed.
E   Calculated: pcc=0.9711294738181017. Required: pcc=0.99.
```

### Logs
- `/tmp/playground_te2_pcc4709.log` — whole-model run showing final assertion (PCC=0.9711, atol=2.0833).
- Per-iteration cumulative sweep logs available locally on branch `kkannan/may21_playground_v2_5_encoder_2_pcc_drop`.

### Expected behavior
- Whole-model PCC ≥ 0.99 on n150 for `playground_v2_5/text_encoder_2` (`CLIPTextModelWithProjection`, fp32 override).

### Related issues
- Pattern matches cumulative-error PCC reports on stacked transformer blocks (e.g. tt-xla#4849, #4328). Likely a small per-block PCC loss in CLIPEncoderLayer (self-attention / MLP / layernorm pipeline) that compounds across 32 iterations — next step is to file an op-level tt-metal issue once the dominant per-layer source is isolated.
