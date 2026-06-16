# FLUX.2 component test — true status on latest main (LLMBox / 8-chip T3K, wh-lb-46)

Date: 2026-06-15. Branch `akannan/bringup_flux2`. skip/xfail markers commented out so true
status is observed. Machine: properly-connected T3000 (8 chips, distinct mesh coords), HF token OK.

## Sharded tests (the relevant ones on 8 chips) — BOTH FAIL with DRAM OOM

| Test | Result | Stage | Detail |
|------|--------|-------|--------|
| `test_text_encoder_sharded` | FAILED | execution — **input tensor prep** | OOM allocating **320 MB** DRAM buffer; DRAM already 1039 MB full (31 MB free). `prepareInputTensor` → `MeshBuffer::create`. `bank_manager.cpp:462` |
| `test_transformer_sharded` | FAILED | execution — **mid-graph `tilize` op** | OOM allocating **162 MB** DRAM output tensor; DRAM 1034 MB full (36 MB free). `ttnn::prim::tilize` → `create_device_tensor`. `bank_manager.cpp:462` |

Both compile end-to-end and reach the runtime; they die during execution because the sharded
weights consume nearly all per-bank DRAM (~1 GB/bank used of 1.07 GB), leaving ~30 MB — too
little for the next large activation/input allocation.

### Change vs the old xfail reasons
- `test_text_encoder_sharded` OLD xfail = "PCC 0.9684 < 0.99" (it used to run to completion).
  NOW it OOMs at input prep before producing output → **regressed / changed failure mode**.
- `test_transformer_sharded` OLD xfail = DRAM OOM (`bank_manager.cpp:462`).
  NOW still DRAM OOM but on a `tilize` op mid-execution (162 MB) rather than the previously
  cited 56 MB buffer → **same class (DRAM-bound), different allocation site**.

## Single-device tests — IGNORED per user (LLMBox is 8-chip)
All three fail with `Device count mismatch: 1 vs 8` (compiled for 1 device, runtime exposes 8) —
expected on an 8-chip box; these are meant for single-chip runners.
- `test_vae_decoder`: FAILED — `Bad StatusOr access: INTERNAL: Error code 13` (`Device count mismatch: 1 vs 8`).
- `test_text_encoder` (single): FAILED — same device-count mismatch.
- `test_transformer` (single): FAILED — same device-count mismatch.

## Logs
`flux_updated_logs/test_{vae_decoder,text_encoder_single,transformer_single,text_encoder_sharded,transformer_sharded}.log`

## Bottom line
On latest main + 8-chip LLMBox, **both sharded FLUX.2 components are DRAM-OOM-bound at execution**
(weights nearly fill DRAM). Next step (not yet done, per user): reduce device-DRAM pressure —
shard/replicate spec review, dtype/memory-config of weights, or activation buffering — to fit the
text encoder input (320 MB) and transformer tilize output (162 MB).
