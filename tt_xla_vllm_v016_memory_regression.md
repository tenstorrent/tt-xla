# vLLM v0.16.0 Host Memory Regression Investigation

## Summary

vLLM v0.16.0 causes ~3x host memory blow-up compared to v0.15.0 during model compilation (dynamo tracing). For Qwen3-1.7B, peak RSS goes from 7.5 GB to 22 GB. The regression scales with model size — for Qwen3-32B it reaches ~435 GB.

## Confirmed Facts

### A/B Comparison (same torch 2.9.1, same torch_xla, same PJRT plugin)

| Metric | v0.15.0 | v0.16.0 |
|--------|---------|---------|
| After model load | 6.73 GB | 6.47 GB |
| During `_dummy_run` forward (dynamo trace) | **5.52 GB** (drops!) | **20.63 GB** (+14 GB) |
| After full init | 5.69 GB | 20.89 GB |
| Peak RSS (1.7B model) | 7.5 GB | 22 GB |
| Peak RSS (0.6B model) | 3.6 GB | 8.5 GB |
| Peak RSS (4B model) | 16.5 GB | 49.6 GB |

### Where the memory goes

- **14.2 GB spike** happens during `self.model(input_ids=..., positions=..., inputs_embeds=...)` inside `_dummy_run`
- Spike occurs BEFORE `extract_graph_helper` (torch_xla backend) is invoked — this is **pure torch._dynamo tracing overhead**
- `extract_graph_helper`, `_get_tensors_xla_device_data_node`, and `GraphInputMatcher` add negligible memory (+0.06 GB)
- Each forward trace creates ~9,343 new XLA tensors totaling ~22 GB logical size
- The largest are weight-shaped duplicates: 336x (6144, 2048) = 8 GB, 252x (2048, 6144) = 6 GB
- Only 1 dynamo compilation per forward (no graph breaks)
- Only 343-398 graph inputs captured by XLA — the 9K tensors are dynamo intermediates

### What does NOT cause it

- `_torch27_patch_tensor_subclasses` removal (PR #33209) — re-adding it has zero effect. Tested both as context manager in decorators.py and as direct config set in model_runner
- `traceable_tensor_subclasses` dynamo config — setting it directly before forward has no effect
- `can_dispatch_torch_function` patch — patching it to return False has no effect
- `compilation/decorators.py` — full v0.15 file swapped into v0.16: no effect
- `compilation/wrapper.py` — full v0.15 file swapped into v0.16: no effect
- `forward_context.py` — full v0.15 file swapped into v0.16: no effect
- `transformers` version (4.57.1 vs 4.57.6) — Qwen3 model code is identical
- TT plugin code — nearly identical between versions (just import path changes)
- `assume_32_bit_indexing` config change (True→False) — inductor-only, we use XLA backend

### Compilation config differences between v0.15 and v0.16

```
v0.15: assume_32_bit_indexing: True
v0.16: assume_32_bit_indexing: False

v0.16 adds: fuse_act_padding: False, fast_moe_cold_start: True, static_all_moe_layers: []
```

## Root Cause Hypothesis

The regression is somewhere in vllm 0.16.0's package but NOT in the individual files we tested (decorators.py, wrapper.py, forward_context.py, model code). The change likely spans multiple files or is in a file we haven't bisected yet.

The qualitative difference is striking: v0.15 RSS **drops** during dynamo tracing (6.73→5.52 GB), suggesting dynamo replaces real weight tensors with lightweight proxies. v0.16 RSS **spikes** (6.47→20.63 GB), suggesting dynamo creates additional copies of weights during tracing.

## Reproduction

```bash
# On kmabee/vllm_mem_explosion_debug branch with memory instrumentation
/usr/bin/time -v pytest --durations=0 -svv --log-memory \
  "tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py::test_tensor_parallel_generation_llmbox_small[Qwen/Qwen3-1.7B-False-False]"

# Check RSS timeline
grep "\[MEM\]" <logfile>
```

## Next Steps

1. Binary search: swap progressively larger chunks of v0.15 vllm into v0.16 to isolate the file(s)
2. Enable `TORCH_LOGS=+dynamo` on both versions and diff the trace output
3. Check if vllm 0.16 changed how model weights are stored (e.g., `nn.Parameter` vs tensor subclass wrapping) which affects how dynamo traces through them
4. File upstream vllm issue with the A/B data

## Key Files

- `integrations/vllm_plugin/vllm_tt/worker.py` — instrumented with `_log_rss()` at key init stages
- `integrations/vllm_plugin/vllm_tt/model_runner.py` — instrumented with RSS logging in `_dummy_run`, `capture_model`, `profile_run`; monkey-patches `extract_graph_helper` and `_get_tensors_xla_device_data_node`
- Log files: `vllm_qwen3_1.7b_mem_profile_*.log`
