# Sampler Perf + Correctness Testing Plan

## Goal

Fast iteration loop for non-greedy sampler changes: catch correctness regressions
in seconds and measure throughput without loading a full model.

## Testing Tiers (fastest to slowest)

### Tier 1: Saved-logits test (~5-30 seconds)

**What**: Load pre-captured logits from a real Llama-3.2-3B decode step,
run just the sampler (no model, no compilation warmup beyond sampler graph),
check output quality + measure throughput.

**Fixture**: `tests/integrations/vllm_plugin/sampling/fixtures/llama3_2_3b_decode_step1.pt`

**Correctness check**: Sampled token IDs must fall within the top-K of the
logit distribution (not garbage tokens with very low probability).

**Throughput**: Time 100 sampler calls, report tok/s.

**Pros**: Real logit distribution, production vocab size (128256), real batch
shapes, benchmarking built-in, no model load.

**TODO**: Reconstruct `perf_debug/test_sampler_throughput.py` (was a scratch
script, never committed). Key modes needed:
- `--device`: run tt::sampling on TT device (default)
- `--cpu`: run CPU reference path (`sample_from_logits_cpu`)
- `--greedy`: run compiled argmax path

Command:
```bash
python perf_debug/test_sampler_throughput.py
python perf_debug/test_sampler_throughput.py --cpu
python perf_debug/test_sampler_throughput.py --greedy
```

### Tier 2: Synthetic sampler test (~1-2 minutes)

**What**: `tests/integrations/vllm_plugin/sampling/test_sampling_params_synthetic.py`

Runs compiled sampler graph on synthetic logits at production vocab sizes
(Llama-3-8B=128256, Qwen3=151936, etc.). No model load. Validates token
is in range and correct graph is compiled.

**When to use**: After Tier 1 passes — broader param coverage (top_k, top_p,
min_p, penalties) across vocab sizes.

Command:
```bash
pytest -svv tests/integrations/vllm_plugin/sampling/test_sampling_params_synthetic.py
```

### Tier 3: CI correctness tests (~15-20 minutes each)

Two tests for catching output corruption:

**a) Nongreedy coherence test** (single_device, push):
```bash
pytest -svv tests/integrations/vllm_plugin/sampling/test_sampling_params.py::test_output_coherence_nongreedy
```
Uses Llama-3.2-1B + opt_level=1. Catches issue #4325 (RoPE/sampler corruption).
Checks: no non-Latin scripts, >60% ASCII alphabetic ratio.

**b) Sampling quality smoke tests** (OPT-125M + Llama-1B push, Llama-3B nightly):
```bash
pytest -svv tests/integrations/vllm_plugin/generative/test_sampling_quality_smoke.py -k "llama_1b"
```
Runs batch=1 and batch=2, releases device before asserting.

### Tier 4: Full generation tests (30+ minutes)

```bash
pytest -svv tests/integrations/vllm_plugin/generative/test_llama3_3b_generation.py
```
Covers opt_level=0 baseline, opt_level=1 batch1/batch2, and trace variant.

## Recommended iteration loop

For each sampler change:
1. Tier 1 (saved logits): quick sanity on correctness + perf delta
2. Tier 2 (synthetic): broader param coverage
3. If both pass: Tier 3a (nongreedy coherence) to catch end-to-end regressions
4. Before PR: Tier 3b + Tier 4

## Known sensitivities

- **opt_level=1 + batch=1 + RoPE models**: most sensitive combination for
  detecting sampling corruption (see issue #4325)
- **top_p enabled**: makes corruption more visible — both batch sizes fail
  when broken, vs only batch=1 without top_p
- **OPT-125M**: does NOT use RoPE, will not catch RoPE-related sampling bugs
- **Stochastic**: a single passing run at opt_level=1 is not sufficient —
  the corruption is intermittent. Run 2-3 times before concluding clean.
