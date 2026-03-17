# Benchmark Tests

Performance benchmarks for TT-XLA across LLMs, vision models, and encoder models. All the tests here are run on nightly and results can be viewed on the [Superset performance dashboard](https://superset.tenstorrent.com/superset/dashboard/p/b3YOz8MaQyD/).

## Quick Start

```bash
# Activate the environment
source venv/activate

# n150 — single-chip models
pytest -svv tests/benchmark/test_llms.py::test_llama_3_2_1b
pytest -svv tests/benchmark/test_vision.py::test_resnet50

# LLMBOX / QB — tensor-parallel models
pytest -svv tests/benchmark/test_llms.py::test_llama_3_1_70b_tp

# Galaxy
pytest -svv tests/benchmark/test_llms.py::test_llama_3_1_70b_tp_galaxy
```

## Useful CLI Options

| Option | Description |
|--------|-------------|
| `--num-layers N` | Override number of model layers (positive integer) |
| `--max-output-tokens N` | Limit generated tokens (LLMs only) (useful for profiling runs) |


### Examples

```bash
# Short profiling run
pytest -svv tests/benchmark/test_llms.py::test_qwen_3_0_6b --num-layers 1 --max-output-tokens 3
```

## Directory Structure

```
tests/benchmark/
├── test_llms.py             # LLM test definitions
├── test_vision.py           # Vision model test definitions
├── test_encoders.py         # Encoder model test definitions
├── benchmarks/              # Core benchmark implementations
├── llm_utils/               # LLM-specific utilities
```

## Profiling

See [PROFILING.md](PROFILING.md) for detailed instructions on device and host profiling with Tracy.

## CI

Performance benchmarks run via the **"Performance benchmark"** GitHub Actions workflow (`manual-benchmark.yml`). It can be triggered from the GitHub UI [here](https://github.com/tenstorrent/tt-xla/actions/workflows/manual-benchmark.yml) or from the CLI with `gh workflow run`.

### Workflow inputs

- **`mlir_override`** — Git SHA of a tt-mlir commit to use instead of the default toolchain. Default: empty.
- **`test_filter`** (Filter tests based on the name property) — Comma-separated substrings to match against test names (case insensitive). E.g. `"llama,qwen"` runs all tests with "llama" or "qwen" in the name. Default: empty (all tests).
- **`runs-on-filter`** (Architecture you want to run the tests on) — Hardware to run on: `n150`, `p150`, `n300-llmbox`, `galaxy-wh-6u`, or `All`. Default: `n150`.
- **`sh-runner`** (Run on shared runner) — Must be set to `false` for precise performance comparison. Shared runners provide less stable results. Default: `true`.
- **`perf_regression_check`** (Enable perf metrics regression testing) — Compare results against the last nightly run and flag >5% regressions in samples/sec. Default: `false`.

### Examples

```bash
# Run llama and qwen benchmarks on n300-llmbox for a specific tt-mlir commit
gh workflow run "Performance benchmark" --ref my-branch \
  -f test_filter="llama,qwen" \
  -f runs-on-filter=n150 \
  -f mlir_override=abc123def
```

### Checking performance for a feature branch

When working on a performance-related change, run CI benchmarks for all model categories your change could affect. Think about whether the change impacts LLMs, vision models, encoders, or all of them, and whether it affects single-chip, multi-chip (TP), or both.

1. **Run the relevant benchmarks on bare metal** with `sh-runner=false` for stable, reproducible numbers. Shared runners have higher variance and are not suitable for precise performance comparison.

2. **Enable regression checking** with `perf_regression_check=true` to automatically compare your results against the last nightly run on main. The check flags any >5% regression in samples/sec.

3. **Manually compare against the nightly dashboard.** The automated regression check is a useful first pass, but you should also review the [Superset performance dashboard](https://superset.tenstorrent.com/superset/dashboard/p/b3YOz8MaQyD/) which tracks nightly main performance over time.

Example: checking a change that affects all single-chip LLMs:

```bash
gh workflow run "Performance benchmark" --ref my-feature-branch \
  -f test_filter="llama,phi,gemma,falcon,qwen,mistral" \
  -f runs-on-filter=n150 \
  -f sh-runner=false \
  -f perf_regression_check=true
```

## Adding a New Test

### Adding a variant of an existing architecture

**Add the test function** in the appropriate test file (e.g. `test_llms.py`):

```python
def test_new_model(
    output_file, num_layers, request, accuracy_testing, batch_size, max_output_tokens
):
    from third_party.tt_forge_models.new_model.causal_lm.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant.NEW_VARIANT
    test_llm(
        ModelLoaderModule=ModelLoader,
        variant=variant,
        output_file=output_file,
        num_layers=num_layers,
        request=request,
        accuracy_testing=accuracy_testing,
        batch_size=batch_size,
        max_output_tokens=max_output_tokens,
    )
```

Vision and encoder tests follow the same pattern — import the existing `ModelLoader`, pick a variant, and call `test_vision()` or `test_encoder()`.

### Adding the test to CI

In order for a new test to run in CI, it needs to be added to `.github/workflows/perf-bench-matrix.json`.
