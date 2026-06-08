### Describe the bug

- **Model key:** `krea_realtime_video/pytorch-RealtimeVideo14B-tensor_parallel-inference` — `CausalWanModel`, the 14B (~14.29B params) Wan video DiT behind `krea/krea-realtime-video`. Component test: `tests/torch/models/krea_realtime/test_transformer.py::test_transformer_sharded`.
- **Observed vs expected:** the tensor-parallel sharded TT path **compiles, runs, and produces a finite output** on n300-llmbox (mesh `(2, 4)` = 8 chips), but **PCC vs the CPU golden is `0.9630219547073653`**, below the required `0.99`. The test is marked `xfail(strict=False)` until the numerics are bit-accurate.
- **Surface error:** `AssertionError: Evaluation result 0 failed: PCC comparison failed. Calculated: pcc=0.9630219547073653. Required: pcc=0.99.` This is a **numerical accuracy gap, not a compile or runtime crash** — the device path is functionally green.
- **Arch / dtype:** `n300-llmbox`, fully bf16 on both golden and TT paths (`dtype_override=torch.bfloat16`). Shard pattern: Megatron-1D on the `model` axis; stem replicated.

### Call chain

```
CausalWanWrapper            # tensors-only forward (x, t, context) -> noise_pred
  → CausalWanModel          # 14B Wan video DiT, ~40 transformer blocks
      → per-block self-attention   (q/k/v column-sharded, o row-sharded on "model")
          → WanRMSNorm(dim=5120) on flat q/k   # causal_model.py:280, full-dim qk-RMSNorm
          → complex128 RoPE (self.freqs)       # "force moved to XLA"
      → per-block FFN              (column → row sharded, all-reduce back to full dim)
      → REPLICATED stem           # patch_embedding, text/time embeddings,
                                  # time_projection, head, norm3, modulation
```

### Key observations

- **Shard spec is structurally valid:** `1014` tensors sharded, `81` replicated, `0` unresolved (no rank / axis / divisibility problems). Megatron-1D `(("model", None) / (None, "model"))` on per-block attention + FFN; column→row attention/FFN all-reduce back to full dim.
- **Stem is intentionally replicated.** Sharding it broke at `time_projection → .unflatten(1, (6, dim))` (`causal_model.py:1085`): `6` is not divisible by the `model` axis (`4`), so Shardy could not propagate the sharding — `reshape.2: number of output elements (92160) doesn't match expected (23040)`. Leaving the stem replicated is what made the run compile.
- **Suspected numerical contributors (need isolation):**
  1. **bf16 accumulation over ~40 blocks** — the entire run is bf16 on both golden and device; cumulative round-off across the depth is the most likely dominant term.
  2. **Full-dim qk-RMSNorm under column sharding** — `WanRMSNorm(dim=5120)` is applied to flat q/k *before* the head-view (`causal_model.py:280`). With q/k column-sharded on `model`, the RMS reduction spans the sharded dim and relies on GSPMD inserting an all-reduce; if variance is computed per-shard this is a systematic per-shard error.
  3. **complex128 RoPE** — `self.freqs` is `complex128` and is "force moved to XLA" (see the runtime warning in the log); the complex `rope_apply` precision on TT is unverified.
- **In-band but not relaxing:** `0.963` sits in the band this repo accepts for other large bf16 models (deepseek / kimi / deepseek_v4 run `0.95–0.98`), but per bringup policy we **keep `required_pcc=0.99` and track the gap** rather than relax the threshold silently.
- The sibling components of this pipeline pass on the same branch: **text_encoder TP PCC ≥ 0.99** (8-chip) and **VAE single_device PCC ≥ 0.99** (p150). Only the transformer is below bar.

### Experiments / sanities

| Stage | Result | Notes |
|-------|--------|-------|
| text_encoder TP verify (8-chip) | PASS (PCC ≥ 0.99) | `logs/iter_2_verify.log` |
| transformer TP first run (smoke, tt-only) | PASS (runs) | `logs/iter_2_transformer_repair.log`, ~1381 s |
| transformer TP verify (cpu golden + sharded TT + PCC) | **FAIL — PCC 0.963** | `logs/iter_3_transformer_verify.log`, ~1214 s |
| VAE single_device verify (p150) | PASS (PCC ≥ 0.99) | `logs/iter_2_vae_verify.log` |

### Steps to reproduce

```bash
# n300-llmbox, 8-chip mesh (boards 0-3). ~20 min: full recompile, no executable cache.
TT_XLA_ARCH=n300-llmbox TT_VISIBLE_DEVICES=0,1,2,3 TT_XLA_SPMD=1 CONVERT_SHLO_TO_SHARDY=1 \
  pytest -svv "tests/torch/models/krea_realtime/test_transformer.py::test_transformer_sharded"
```

Captured I/O (480x832, 3 latent frames):

```
x        [1, 16, 3, 60, 104]  bf16
t        [1, 3]               float32
context  [1, 512, 4096]       bf16
OUT:     [1, 16, 3, 60, 104]  bf16
```

Failing excerpt:

```
tests/infra/evaluators/comparison_evaluator.py:73: in evaluate
    ComparisonEvaluator._assert_on_results(_comparison_result)
tests/infra/evaluators/evaluator.py:72: in _assert_on_results
    assert False, "\n".join(error_messages)
E   AssertionError: Evaluation result 0 failed: PCC comparison failed.
    Calculated: pcc=0.9630219547073653. Required: pcc=0.99.
```

### Logs

- Primary: `/proj_sw/user_dev/ctr-akannan/2_jun_yyz/tt-xla/.claude/bringup/krea_realtime_video/logs/iter_3_transformer_verify.log` — **line 70**: the PCC assertion (`pcc=0.9630219547073653`, required `0.99`). Line 21–39 shows the `complex128` RoPE `self.freqs` tensor being force-moved to XLA. Line 54: `Created device mesh: (2, 4) with 8 devices.`
- Smoke run (tt-only, passes): `.claude/bringup/krea_realtime_video/logs/iter_2_transformer_repair.log`
- State / history: `.claude/bringup/krea_realtime_video/state.json`, `bringup_steps.txt`

### Expected behavior

The sharded TT transformer output should match the CPU golden to `PCC ≥ 0.99`, consistent with the sibling text_encoder (TP) and VAE (single_device) components of this pipeline. The current `0.963` indicates accumulated numerical error in the device path that should be isolated and reduced rather than absorbed by a lower threshold.

### Suggested next steps

1. **A/B the qk-norm sharding:** replicate `norm_q` / `norm_k` (force exact normalization via all-gather) vs the current `("model",)` sharding, and compare PCC. This isolates whether the per-shard RMS variance is a systematic contributor.
2. **Bound the bf16 contribution:** run an fp32 golden / compute path for the attention norm + RoPE and re-measure PCC, to separate cumulative bf16 round-off (over ~40 blocks) from a structural sharding error.
3. **Verify the complex128 RoPE path on TT:** confirm `rope_apply` keeps precision after `self.freqs` is force-moved to XLA; consider a real-valued (cos/sin) RoPE formulation if the complex path loses precision.

### Related issues

- `#4462` (open) — **Model Bringup: Krea Realtime 14B** — parent tracking task; link this issue under it.
- `#4465` (open) — **[Krea Realtime 14B] VAE decoder: Value out of range** — sibling component of the same pipeline (now fixed via causal-slice monkey-patch; tracks a different failure class).
- `#4464` (closed) — **[Krea 14B] transformer: Failed to create zero attribute** — prior transformer bringup blocker for the same model (resolved); the CUDA-hardcoded sinusoidal-embedding patch referenced in the test docstring stems from this.
- `#3587` (open) — **test_models on `lb-blackhole`** — multichip / tensor-parallel bringup tracking.

### Notes

- **Arch:** n300-llmbox, mesh `(2, 4)` = 8 chips (`TT_VISIBLE_DEVICES=0,1,2,3`).
- **Classification:** model-level numerical accuracy (PCC), not an op-level compile/runtime crash. The device path is functionally green; this tracks closing the bf16/sharding numerics gap.
- **Filing:** tt-xla needs **Type: Bug** set in the GitHub UI (the gh CLI cannot set it). Suggested labels: `bug`, plus `model-bringup` / `tensor-parallel` if those labels exist in the repo.
