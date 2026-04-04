---
name: vLLM Support Workflow
description: End-to-end guide for onboarding any hardware tier to the vLLM integration
test suite in tt-xla and wiring it into CI. Covers n300-llmbox, galaxy-wh-6u, and future
hardware. Reference PR: tenstorrent/tt-xla#3814.
---

## Hardware Tiers

| Runner label     | Hardware             | Pytest marker   | Chip count | `use_2d_mesh` |
|------------------|----------------------|-----------------|------------|---------------|
| `n150`           | Single Wormhole      | `single_device` | 1          | `False`       |
| `n300`           | Dual Wormhole        | `dual_chip`     | 2          | `False`       |
| `n300-llmbox`    | 4× n300 LLMBox       | `llmbox`        | 8          | `False`       |
| `galaxy-wh-6u`   | Galaxy WH 6U         | `galaxy_wh_6u`  | 32+        | `True`        |

> **Marker convention:** derive from the runner label — replace `-` with `_`
> (e.g. `n300-llmbox` → `llmbox`, `galaxy-wh-6u` → `galaxy_wh_6u`).

---

## End-to-End Workflow

Replace `<marker>`, `<runner-label>`, and `<hw-name>` with the values from the table above.

### Step 1 — Add pytest markers and test functions

File: `tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py`

#### Push (smoke) test — validates TP wiring on the new hardware

```python
@pytest.mark.push
@pytest.mark.tensor_parallel
@pytest.mark.<marker>               # e.g. llmbox, galaxy_wh_6u
@pytest.mark.parametrize(
    ["model_name", "enable_const_eval"],
    [
        pytest.param("<small-validated-model>", False),
    ],
)
def test_tensor_parallel_generation_<marker>_small(
    model_name: str,
    enable_const_eval: bool,
):
    llm_args = {
        "enable_tensor_parallel": True,
        "use_2d_mesh": <True|False>,    # True for galaxy-wh-6u; False for n300-llmbox
    }
    check_host_memory(model_name)
    ...
```

#### Nightly test — large frontier models

> **Only add a model here once it has successfully run on the target machine.**
> Start with a single validated model and expand the list incrementally as more are confirmed.

```python
@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.<marker>
@pytest.mark.parametrize(
    ["model_name", "enable_const_eval"],
    [
        # Add entries only after a successful run on <hw-name> has been observed.
        pytest.param("<first-validated-large-model>", False),
    ],
)
def test_tensor_parallel_generation_<marker>_large(
    model_name: str,
    enable_const_eval: bool,
):
    llm_args = {
        "enable_tensor_parallel": True,
        "use_2d_mesh": <True|False>,
        "experimental_weight_dtype": "bfp8",   # for models >70B to fit in device memory
    }
    check_host_memory(model_name)
    ...
```

#### Per-machine reference

| Machine         | `use_2d_mesh` | Smoke model (example)                        | Large model threshold |
|-----------------|---------------|----------------------------------------------|-----------------------|
| `n300-llmbox`   | `False`       | `mistralai/Mistral-7B-Instruct-v0.3`         | >30B → use `bfp8`     |
| `galaxy-wh-6u`  | `True`        | `mistralai/Mistral-Large-Instruct-2411`      | >70B → use `bfp8`     |

#### Key `llm_args` fields

| Field | Value | When to include |
|-------|-------|-----------------|
| `enable_tensor_parallel` | `True` | Always, for all multi-chip TP hardware |
| `use_2d_mesh` | `True` | Galaxy and any 2D-mesh topology; omit for llmbox/n300 |
| `experimental_weight_dtype` | `"bfp8"` | Nightly only, for models that won't fit in BF16 |
| `enable_const_eval` | `True` | Very large models (e.g. Llama-3.1-405B) to reduce compile overhead |

> **Do not include `experimental_weight_dtype` in the push/smoke test** — unused parameters
> are a merge blocker (flagged in tenstorrent/tt-xla#3814).

---

### Step 2 — Register the hardware job in the CI test matrix

There are **two separate matrix files** — one for push (smoke) runs and one for nightly runs.
Both use the same field schema.

#### Push matrix

File: `.github/workflows/test-matrix-presets/vllm-model-tests.json`

```json
{
  "runs-on": "<runner-label>",
  "name": "run_vllm_<runner-label>_tests",
  "dir": "./tests/integrations/vllm_plugin",
  "test-mark": "push and tensor_parallel and <marker>"
}
```

#### Nightly matrix

File: `.github/workflows/test-matrix-presets/vllm-model-tests-nightly.json`

```json
{
  "runs-on": "<runner-label>",
  "name": "run_vllm_<runner-label>_tests",
  "dir": "./tests/integrations/vllm_plugin",
  "test-mark": "nightly and tensor_parallel and <marker>"
}
```

**Concrete examples (galaxy-wh-6u):**

```json
// vllm-model-tests.json (push)
{ "runs-on": "galaxy-wh-6u", "name": "run_vllm_galaxy_wh_6u_tests", "dir": "./tests/integrations/vllm_plugin", "test-mark": "push and tensor_parallel and galaxy_wh_6u" }

// vllm-model-tests-nightly.json (nightly)
{ "runs-on": "galaxy-wh-6u", "name": "run_vllm_galaxy_wh_6u_tests", "dir": "./tests/integrations/vllm_plugin", "test-mark": "nightly and tensor_parallel and galaxy_wh_6u" }
```

**Fields:**

| Field | Description |
|-------|-------------|
| `runs-on` | GitHub Actions runner label (must exist in the runner pool) |
| `name` | Job name shown in GitHub Actions — use `run_vllm_<runner-label>_tests` |
| `dir` | Always `./tests/integrations/vllm_plugin` |
| `test-mark` | Pytest `-m` expression — use `push` for smoke matrix, `nightly` for nightly matrix |

> `test-mark` is a boolean expression. Adding the CI entry without a matching pytest test
> causes 0-test-collection and a CI failure.

---

### Step 3 — Verify and open a PR

1. **Local smoke run** (requires access to the target hardware):
   ```bash
   # push smoke
   pytest tests/integrations/vllm_plugin -m "push and tensor_parallel and <marker>" -v

   # nightly
   pytest tests/integrations/vllm_plugin -m "nightly and tensor_parallel and <marker>" -v
   ```

2. **Confirm the runner label** exists in the GitHub Actions runner pool — ask infra if unsure.

3. Open the PR. Both matrix jobs (`vllm-model-tests.json` and `vllm-model-tests-nightly.json`) trigger automatically.

---

## Checklist for adding a new hardware tier

- [ ] Pick the pytest marker (runner label, dashes → underscores)
- [ ] Confirm the runner label exists in the GitHub Actions runner pool
- [ ] Determine `use_2d_mesh` for this hardware (see table above)
- [ ] Add push smoke test: `@pytest.mark.push + tensor_parallel + <marker>`
  - Use a small, fast model
  - Only include `llm_args` fields that are actually used
  - Do **not** add `experimental_weight_dtype` here
- [ ] Add nightly test: only models confirmed to run successfully on this hardware
  - Start with one validated model; add more as they are verified
  - Set `experimental_weight_dtype: "bfp8"` for models that won't fit in BF16
- [ ] Add push entry to `vllm-model-tests.json` (`test-mark: "push and tensor_parallel and <marker>"`)
- [ ] Add nightly entry to `vllm-model-tests-nightly.json` (`test-mark: "nightly and tensor_parallel and <marker>"`)
- [ ] Verify both CI jobs collect ≥1 test (no 0-collection failure)
- [ ] Open PR; tag infra if runner availability is uncertain

---

## Known issues / lessons learned (from tenstorrent/tt-xla#3814)

| Issue | Resolution |
|-------|-----------|
| Unused `experimental_weight_dtype` in push test | Keep it in nightly only; remove from push test signature and `llm_args` |
| CI job collects 0 tests → merge failure | `test_markers` in JSON must match at least one `@pytest.mark.push` test |
| Marker name mismatch (e.g. `galaxy` vs `galaxy_wh_6u`) | Always derive from runner label: replace `-` with `_` |
| Adding unvalidated models to nightly | Only add a model after a successful run on the target hardware is observed |
