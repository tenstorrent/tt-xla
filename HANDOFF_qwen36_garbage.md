# Handoff v2: Qwen3.6-27B garbage output on Tenstorrent (tt-xla + vLLM)

Date: 2026-07-02. Supersedes `HANDOFF_qwen36_garbage.md` (adds this session's
sharding-harness results). Everything below is established by MEASUREMENT.
Where something is inference or untested, it is labeled as such.

## The problem
`Qwen/Qwen3.6-27B` run through tt-xla's vLLM integration (`vllm_tt`) emits
degenerate output. Even a single greedy prefill token (max_tokens=1) is wrong.
- Prompt: `"Continue in English: I like taking walks in the"` — 10 tokens, no BOS:
  `[22791,303,6163,25,353,1040,4483,21780,303,279]`; padded to 32 in the device buffer.
- Config: `enable_tensor_parallel=True, use_2d_mesh=True` → 8 devices, **2×4 mesh**
  (`("batch","model")`), `max_model_len=32, min_context_len=32, max_num_seqs=1`,
  `gpu_memory_utilization=0.2`.
- Model: `Qwen3_5ForConditionalGeneration` (dense, NOT MoE), 64 layers =
  48 GatedDeltaNet (linear-attn) + 16 full-attention (SDPA), hidden 5120,
  head_dim 256, partial_rotary 0.25.
- The vLLM upstream used GPU kernels, so `vllm_tt` REIMPLEMENTS GDN / conv1d /
  l2norm / rope / RMSNorm in pure torch (`integrations/vllm_plugin/vllm_tt/layers/`).

## Tensor-parallel mechanism (confirmed by reading the code)
This is torch_xla **SPMD single-controller**, NOT vLLM multi-process TP:
- `worker.py` forces `parallel_config.tensor_parallel_size = 1` when SPMD is on
  (`use_spmd = enable_tensor_parallel or enable_data_parallel`). So the layer code
  runs with `tp_size == 1`; the per-head slicing in GDN `_input_projection` is a
  no-op divide-by-1.
- Sharding is applied purely via `xs.mark_sharding` annotations on the full tensors,
  against a 2×4 `("batch","model")` mesh, in `vllm_distributed_utils.py::shard_model`
  (dispatched by module type) plus activation marks in `model_runner.py`.
- `CONVERT_SHLO_TO_SHARDY=1` → the frontend emits Shardy (`sdy`) annotations;
  tt-mlir lowers `sdy.manual_computation` → TTNN collectives (CCLs).
- IMPLICATION: because `tp_size==1`, the eager-CPU replay (below) runs the SAME
  python as the device. The ONLY thing the device adds on top of the replay is the
  SPMD sharding annotations + their partitioning into CCLs.

## What is DEFINITIVELY established (method, not inference)
1. **Prefill itself strays** — greedy prefill emits garbage; not a decode/sampling-only bug.
2. **Device prefill logits are grossly wrong vs source HF.** Method: dumped device
   top-20 logprobs (`SamplingParams(logprobs=20)`) vs an fp32 CPU HuggingFace forward
   of the same token ids.
   - HF top-1 `" park"` (lp −0.76), then countryside/morning/forest/evening — coherent.
   - Device top-1: `" the"` p≈0.996 (mp=on) — the last input token; or `""` id 225
     (mp=off). **0/20 overlap** with HF either way.
3. **The stray is in the BACKBONE, not lm_head.** Method: captured device pre-lm_head
   hidden (post-final-RMSNorm, last real token) vs HF.
   - cosine(device, HF) = **0.096** (orthogonal). Norms 134.2 vs 139.35 (MATCH) →
     a direction/content scramble, NOT a magnitude collapse.
4. **The vllm_tt MODEL CODE + WEIGHT LOADING are CORRECT.** *** Key result. *** Method:
   eager-CPU replay — the exact loaded model object, all params/buffers moved to CPU,
   `model.forward()` run eagerly (dynamo disabled, custom ops on CPU paths) with the
   real captured inputs/metadata, compared to HF and to the device.
   - cosine(replay, HF) = **0.9999**, top-20 overlap **20/20**, top-1 `" park"` == HF.
   - cosine(replay, device) = **0.097** (orthogonal).
   → Same code, same loaded weights, run eagerly, reproduce the source model exactly.
5. **⇒ The bug is introduced by the COMPILE/LOWERING path**, not model code or weights.
6. **Precision is NOT the cause.** Replay was fp32, device bf16, but 0.097 is a gross
   stray; a bf16 replay would still be ~0.99.
7. **chisel is structurally blind to this.** chisel compares each TTNN op
   device-vs-its-own-CPU-golden (confirmed device == CPU-of-the-compiled-graph). It does
   NOT compare the compiled graph to the SOURCE model, so a graph that strayed during
   capture/lowering, or a wrong StableHLO→TTNN op selection, passes chisel. The earlier
   "48 numerics_fail" were PCC-on-near-constant artifacts, not real.

## Sharding-lowering results — THIS SESSION (measured on 8-device 2×4 mesh)
Built a golden-PCC harness: `tests/integrations/vllm_plugin/gdn/test_gdn_sharding.py`.
It runs the REAL vllm_tt ops through `infra.run_graph_test`, which computes an
**unsharded CPU golden** and a **TT-device** result and PCC-compares. Because these ops
are head-parallel / the TP scheme is standard, the sharded device result MUST equal the
unsharded golden; any gross PCC drop = the sharding/CCL lowering scrambled content. A
`replicated` (mesh present, no input sharding, no CCLs) control runs beside each
`sharded` case. Cases: `small` and `qwen3_5` (realistic dims), dtypes fp32 + bf16.
Thresholds: fp32 ≥ 0.99, bf16 ≥ 0.97 (gross-error detectors, not precision tests).

Results — **ALL PASS** (24/24):
- **GDN core delta rule** (`tt_chunk_gated_delta_rule`, prefill), head-sharded on the
  "model" axis (q/k/v/g/beta on head dim; initial_state on its head dim): PASS, all
  dtypes/sizes incl. `qwen3_5` (H16/HV32/head_dim128).
- **in_proj** — merged column-parallel linear (weights `("model","batch")`, faithful to
  `XlaMergedColumnParallelLinear.forward`): PASS.
- **out_proj** — RowParallel linear with input contraction sharded on "model" so the
  matmul REDUCES over "model" (**all-reduce**), weight `("batch","model")` per
  `partition_row_parallel_linear`: PASS.

⇒ **Every sharded GDN-adjacent op tested IN ISOLATION lowers correctly**, including the
out_proj all-reduce — the prime "matching-norm, orthogonal-hidden" suspect. So the bug is
NOT a single sharded op's lowering.

## Ruled OUT (by measurement)
- lm_head / compute_logits / logits-gather-position / sampling (the hidden fed in is
  already wrong).
- Final RMSNorm + its weight (hidden norm magnitude is correct).
- Model math: GDN chunk-vs-recurrent equivalence (to ~1e-8), qkv split/order, gating,
  conv1d, l2norm, mrope (identity for text), rope, output-gate — verified by inspection
  AND by the eager replay matching HF.
- Weight values as loaded (eager replay with those weights == HF).
- bf16 precision compounding.
- **In-isolation lowering of**: head-sharded GDN core delta rule; column-parallel in_proj;
  RowParallel out_proj all-reduce (this session, above).

## STILL OPEN / UNTESTED (candidates for the bug)
Per-op isolation cannot see these — they are where the bug most likely lives now:
1. **Whole-graph sharding propagation / composition**: residual-stream sharding across the
   64-layer stack; the sharding-constraint forward hooks; any resharding inserted where an
   op's input arrives with a different spec than the isolated test assumed. NOT exercised
   by single-op harness tests.
2. **The 16 full-attention layers' TTNN SDPA, sharded.** The eager replay substituted
   `F.scaled_dot_product_attention` (the CPU path of `tt.scaled_dot_product_attention`),
   NOT the TTNN SDPA kernel. So TTNN SDPA semantics + its sharded lowering are untested.
3. **Frontend capture**: dynamo FX capture specialization; torch_xla → StableHLO;
   StableHLO → TTNN op selection. Not separated from the tt-mlir backend yet.
4. mp=on vs mp=off produce DIFFERENT garbage (" the" vs id-225). Both are garbage; it is
   NOT proven they are the same underlying bug. All hidden/replay analysis used mp=off
   (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, required for in-process hooks).
5. Nothing proves the bug is a single defect.

## Recommended next step (highest information)
**Split frontend vs backend**: run tt-xla's dumped prefill StableHLO on plain XLA-CPU
(jax/xla, not tt). Correct there ⇒ bug is StableHLO→TTNN (tt-mlir backend); wrong ⇒
dynamo / torch_xla→StableHLO (frontend). SHLO dumps exist (`logs/fixed.log`;
`module_builder.cc` dumps them). This is favored now that per-op sharding lowering is
largely cleared.
Secondary: extend the harness to sharded TTNN SDPA (item 2); diff the dynamo-captured
aten FX graph vs the eager python the replay proved correct (item 3).

## How to reproduce / key artifacts
- Device + eager-CPU replay (~20 min compile, in-process):
  `VLLM_ENABLE_V1_MULTIPROCESSING=0 python -m pytest -svv tests/integrations/vllm_plugin/generative/test_prefill_logits.py`
  then `python tmp/compare_replay.py` and `python tmp/compare_hidden.py`.
  The replay in `test_prefill_logits.py` is a reusable ORACLE: eager-CPU vllm_tt ==
  ground truth. (Note: `XlaMergedColumnParallelLinear` stores weights in plain lists
  invisible to `named_parameters` — the replay rebuilds those explicitly.)
- Sharding harness (needs 8 TT devices; runs per-case in seconds–1min):
  `python -m pytest -svv tests/integrations/vllm_plugin/gdn/test_gdn_sharding.py`
- HF fp32 reference: `tmp/hf_reference_logits.py` (logits), `tmp/hf_hidden.py` (hidden).
- Data: `tmp/{device_prefill_logits.json, device_hidden.npy, hf_reference_logits.json,
  hf_hidden.npy, replay_hidden.npy, replay_logits.json}`.
- Full memory trail: `~/.claude/.../memory/chisel-vllm-numerics-debug.md`
  (UPDATES 1–15) and `gdn-decode-graphbreak.md`.

## Bottom line
Model code + weights are proven correct (eager-CPU == source HF at cos 0.9999); the
device-compiled path is orthogonal garbage (cos 0.097) → the bug is in the
compile/lowering. This session cleared the per-op sharding lowering for the GDN core and
its bracketing linears (incl. the out_proj all-reduce) — all 24 harness cases pass. The
bug therefore lives in what per-op isolation cannot see: whole-graph sharding propagation,
the untested TTNN SDPA (sharded), or the frontend capture. Do the frontend-vs-backend
StableHLO-on-XLA-CPU split next.
