# DeepSeek-V2-Lite MLA e2e — root causes & fixes

Bringing up end-to-end generation for DeepSeek-V2-Lite on the TT MLA attention
backend (`tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py`)
surfaced a cascade of bugs. Each was uncovered only after fixing the previous one.

## Prefill (`test_tensor_parallel_mla_prefill_only`)

**1. MoE got a 3D tensor — `ValueError: too many values to unpack`** (`model_runner.py:249`)
DeepSeek uses vLLM-native forwards whose MoE block does `num_tokens, hidden_dim = hidden_states.shape` (expects 2D), but the runner defaults to a `(batch, seq, hidden)` layout. Enabled `use_flat_model_io` for MLA models so the model is called with a flat token stream.

**2. `SharedFusedMoE` bypassed the TT shim → XLA functionalization crash** (`moe_shims.py`)
DeepSeek instantiates `SharedFusedMoE` (shared experts), but only `FusedMoE` was OOT-registered, so the routed path fell back to vLLM's `moe_forward_shared` custom op which doesn't functionalize on XLA. Added `TTSharedFusedMoE` (reusing the dense-bmm `forward_native`, forcing the non-overlapped path).

**3. MoE shims registered too late** (`model_runner.py:1772`)
`moe_shims` was imported *after* the model was built, so the `@CustomOp.register_oot` decorators never applied. Moved the import to the start of `load_model`.

**4. OOM from inflated batched-token shapes** (`platform.py:358`)
The MLA block forced `max_num_batched_tokens = max(max_model_len, 2048)`, discarding the test's `32` and inflating padded shapes. Changed it to preserve the configured value while still guaranteeing `>= max_model_len`.

**5. Expert weights replicated, not sharded → OOM** (`vllm_distributed_utils.py:336`)
The 704 MB allocation matched exactly one layer's stacked `w13` expert weight `[64, 2816, 2048]`. The sharding dispatch table matched by class name and lacked `SharedFusedMoE`/`TTSharedFusedMoE`, so DeepSeek's experts were replicated on all 8 chips. Added both entries.

Result: prompt "I like taking walks in the" → `" woods"`, test passes.

All changes are scoped to the MLA/MoE path (fixes 1, 4, 5 are MLA/MoE-gated; 2 only adds a new class; 3 moves an import earlier and is harmless for non-MoE models), so non-MLA tests should be unaffected.

## Prefill + decode (`test_tensor_parallel_mla_prefill_decode`)

Added a prefill+decode test (DeepSeek-V2-Lite, `max_tokens=11` = 1 prefill token + 10 decode steps), which exercised — and required fixing — the decode path.

**6. Decode misrouted to the prefill kernel** (`model_runner.py:1857`)
The decode step (S=1) hit `flash_mla_prefill`, which asserts seq-len % 32, crashing with "query sequence length must be divisible by 32 but got 1". Root cause: `_attention_layer_names` was built only from `Attention` layers via `get_layers_from_vllm_config(..., Attention)`, but `MLAAttention` is a *separate* class (both derive from `AttentionLayerBase`, neither subclasses the other). So the per-layer `attn_metadata` dict omitted the MLA layers, and `TTMLAAttention.forward` read back `None` — which `_infer_is_prefill` treats as a profiling run and forces to the prefill path. Fixed by collecting both `Attention` and `MLAAttention` layer names into the dict.

Diagnosed by temporarily instrumenting `_infer_is_prefill`, which showed the decode forward (`qshape=(1,16,128)`) arriving with `md_none=True`; the instrumentation was removed after the fix.

Result: prompt "I like taking walks in the" → `" woods. I like to go to the beach. I"` (coherent, 11 tokens), test passes.
