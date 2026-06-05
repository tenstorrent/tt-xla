# Issue draft — file with: gh issue create --repo tenstorrent/tt-xla

**Title:** [krea-realtime-video] CausalWanModel transformer sharded TP: PCC 0.963 < 0.99 on n300-llmbox

**Labels:** bug, model-bringup, tensor-parallel

**Body:**

## Summary
The Krea Realtime Video 14B transformer (`CausalWanModel`, custom Wan video DiT)
runs end-to-end as a tensor-parallel sharded model on n300-llmbox (mesh (2,4) =
8 chips), but its output **PCC vs the CPU golden is 0.963**, below the 0.99 bar.
The test is marked `xfail` until the numerics are bit-accurate.

- Test: `tests/torch/models/krea_realtime/test_transformer.py::test_transformer_sharded`
- Loader: `third_party/tt_forge_models/krea_realtime_video/pytorch/` (shard spec
  `shard_transformer_specs`, Megatron-1D on the `model` axis; stem replicated)
- Env: `TT_XLA_ARCH=n300-llmbox TT_VISIBLE_DEVICES=0,1,2,3 TT_XLA_SPMD=1 CONVERT_SHLO_TO_SHARDY=1`

## Repro
```bash
TT_XLA_ARCH=n300-llmbox TT_VISIBLE_DEVICES=0,1,2,3 TT_XLA_SPMD=1 CONVERT_SHLO_TO_SHARDY=1 \
  pytest -svv tests/torch/models/krea_realtime/test_transformer.py::test_transformer_sharded
```
Result: `AssertionError: PCC comparison failed. Calculated: pcc=0.9630219547073653. Required: pcc=0.99.`
(1 run = ~20 min; full recompile, no executable cache.)

## Suspected contributors (need isolation)
1. **bf16 accumulation** over 40 transformer blocks (run is fully bf16 on both
   golden and TT).
2. **Full-dim qk-RMSNorm under column sharding** — `causal_model.py:280`
   applies `WanRMSNorm(dim=5120)` to the flat q/k **before** the head-view; with
   q/k column-sharded on `model`, the RMS reduction spans the sharded dim and
   relies on GSPMD inserting an all-reduce. If the variance is computed per-shard
   this introduces a systematic per-shard error.
3. **complex128 RoPE** — `self.freqs` is complex128 and is "force moved to XLA";
   verify the complex rope_apply path keeps precision on TT.

## Notes
- The shard spec itself is structurally valid (1014/1095 params sharded, 0
  rank/axis/divisibility problems; column→row attention/FFN all-reduce back to
  full dim; stem replicated to avoid the `time_projection -> unflatten(1,(6,dim))`
  reshape that 6∤4 cannot propagate).
- 0.963 is in the band this repo accepts for other large bf16 models
  (deepseek/kimi/deepseek_v4 use 0.95–0.98), but per bringup policy we keep
  0.99 and track the gap rather than relax silently.

## Suggested next steps
- A/B the qk-norm: replicate `norm_q`/`norm_k` (force exact normalization via
  all-gather) vs current `("model",)` sharding; compare PCC.
- Try an fp32 golden/compute path for the attention norm + RoPE to bound the
  bf16 contribution.
