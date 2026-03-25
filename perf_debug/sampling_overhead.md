# Sampling Overhead Debug: Device vs CPU Sampling

## Problem

Non-greedy sampling on device is significantly slower than CPU sampling for vLLM on Wormhole single-chip. The overhead is large enough to cut throughput in half regardless of model size, confirming the bottleneck is in the compiled sampling graph itself.

## Setup

- Hardware: Wormhole single-chip
- vLLM direct LLM mode (no server overhead)
- batch_size=1, max_model_len=2048 (8B) / 1024 (OPT)
- 128 output tokens per prompt

## Reproduce

```bash
# OPT-125M (fastest iteration — model compute is negligible, isolates sampling cost)
python examples/vllm/opt-125m/chat.py --benchmark --temperature 0.8
python examples/vllm/opt-125m/chat.py --benchmark --temperature 0.8 --cpu-sampling
python examples/vllm/opt-125m/chat.py --benchmark --temperature 0.0

# Llama-3.1-8B-Instruct (production model, ~8min compile)
python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --temperature 0.8
python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --temperature 0.8 --cpu-sampling
python examples/vllm/Llama-3.1-8B-Instruct/chat.py --benchmark --temperature 0.0
```

## Results

### OPT-125M (vocab_size=50272)

| Configuration | tok/s | ms/token | Sampling overhead |
|---|---|---|---|
| Greedy device (temp=0.0) | 11.61 | 86ms | baseline |
| Non-greedy CPU (temp=0.8, top_p=0.9) | 11.00 | 91ms | +5ms (negligible) |
| Non-greedy device (temp=0.8, top_p=0.9) | 5.94 | 168ms | **+82ms** |

### Llama-3.1-8B-Instruct (vocab_size=128256)

| Configuration | tok/s | ms/token | Sampling overhead |
|---|---|---|---|
| Greedy device (temp=0.0) | 9.82 | 102ms | baseline |
| Non-greedy CPU (temp=0.8, top_p=0.9) | 7.89 | 127ms | +25ms |
| Non-greedy device (temp=0.8, top_p=0.9) | 4.02 | 249ms | **+147ms** |

### Analysis

- **Non-greedy device sampling adds 82-147ms per token.** This is the dominant bottleneck — it nearly doubles per-token latency on both models.
- **Greedy device is fast.** The greedy compiled path is efficient on both models.
- **CPU non-greedy is nearly free.** On OPT-125M it adds only 5ms; on 8B it adds 25ms (likely logit transfer cost for larger vocab).
- **Overhead scales with vocab size.** 82ms at vocab 50K vs 147ms at vocab 128K suggests the cost is O(vocab_size) — likely in sort/top-k or softmax over the full vocabulary.
- **Model size doesn't matter.** OPT-125M (125M params) and Llama-8B show similar non-greedy device overhead, confirming the bottleneck is purely in the sampling graph.

## Compilation Cost

Engine init times (profile + KV cache + warmup):
- OPT-125M: ~100s
- Llama-3.1-8B: ~467s

## What's in the Sampling Graph

The device sampling path (`cpu_sampling=False`) compiles via `torch.compile(backend="tt")`:
- `integrations/vllm_plugin/vllm_tt/model_runner.py`: `sample_from_logits()` (line ~2131)
- `integrations/vllm_plugin/vllm_tt/sampler.py`: `Sampler` module with XLA-friendly ops
- Operations: temperature scaling, top-k/top-p masking, softmax, multinomial sampling

The CPU path (`cpu_sampling=True`) does all of the above on CPU after pulling logits from device:
- `integrations/vllm_plugin/vllm_tt/model_runner.py`: `sample_from_logits_cpu()` (line ~2152)

## Hypotheses

1. **Sort/top-k over full vocab is slow on device.** The non-greedy path sorts or does top-k over vocab_size elements. Known issue with `ttnn.sort` for large tensors. The overhead scaling with vocab size (82ms at 50K vs 147ms at 128K) supports this.
2. **Multinomial sampling is expensive on device.** Random number generation + multinomial may not map well to Wormhole.
3. **Multiple device-host sync points.** The sampling graph may force synchronization (e.g. checking top-p cumulative sum thresholds) that serializes execution.
4. **Graph fragmentation.** The sampling graph may not fuse well, causing many small kernel launches vs one efficient CPU pass.

## Next Steps

- [ ] Profile the OPT-125M non-greedy device sampling graph to identify slow ops
- [ ] Check if top-k/top-p sort is the dominant cost (try temp>0 without top-p)
- [ ] Check if the sampling graph triggers multiple compilations or retraces
- [ ] Run Llama-3.2-1B to get a third vocab-size data point
- [ ] Investigate whether a hybrid approach (model on device, sampling on CPU) is viable as default

## Log Files

- `perf_debug/llama3.1_8b_non_greedy_device.log`
- `perf_debug/llama3.1_8b_non_greedy_cpu.log`
- `perf_debug/llama3.1_8b_greedy_device.log`
- `perf_debug/opt125m_non_greedy_device.log`
- `perf_debug/opt125m_non_greedy_cpu.log`
- `perf_debug/opt125m_greedy_device.log`
