---
name: vllm-model-bringup
description: Generates a vLLM bringup test file for a given model and target device (single/n150/p150, n300-llmbox, galaxy). Use when the user wants to bring up a new model on vLLM for Tenstorrent hardware.
allowed-tools: Read Grep Glob Write Bash Task
---

# vLLM Model Bringup

Generates a bringup test file for a new vLLM model on Tenstorrent hardware.

## How It Works

When invoked, collect these inputs from the user's message (ask if `model` or `device` are missing; `input` is optional):

- **`model`** — HuggingFace model ID (e.g. `meta-llama/Llama-3.2-3B`)
- **`device`** — One of: `single`, `n150`, `p150`, `n300-llmbox`, `galaxy`
- **`input`** *(optional)* — Custom prompt string for `prompts = [...]`.

---

## Device → Marker Mapping

| Device | pytest markers |
|--------|---------------|
| `single`, `n150`, `p150` | `@pytest.mark.single_device` |
| `n300-llmbox` | `@pytest.mark.tensor_parallel` + `@pytest.mark.llmbox` |
| `galaxy` | `@pytest.mark.tensor_parallel` + `@pytest.mark.galaxy` |

---

## Behavior by device and input

| Device | `input` given | Action |
|--------|--------------|--------|
| `single` / `n150` / `p150` | yes or no | Append new standalone function to `test_llama3_3b_generation.py`. Use `{input}` or default `"I like taking walks in the"` |
| `n300-llmbox` / `galaxy` | **yes** | Append new standalone function to `test_tensor_parallel_generation.py` with custom prompt |
| `n300-llmbox` / `galaxy` | **no** | Insert a new `pytest.param` entry into the existing `test_tensor_parallel_generation_llmbox_large` parametrize list in `test_tensor_parallel_generation.py` |

---

## Templates

### Single device — new standalone function

```python
@pytest.mark.nightly
@pytest.mark.single_device
def test_{snake_name}_generation():
    prompts = [
        "{input}",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": "{model}",
        "max_num_batched_tokens": 128,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")
```

### n300-llmbox — new standalone function (input provided)

```python
@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize("use_2d_mesh", [True, False])
def test_{snake_name}_generation_n300_llmbox(use_2d_mesh: bool):
    prompts = [
        "{input}",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": "{model}",
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "use_2d_mesh": use_2d_mesh,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")

    check_host_memory("{model}")
```

### galaxy — new standalone function (input provided)

```python
@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.galaxy
@pytest.mark.parametrize("use_2d_mesh", [True, False])
def test_{snake_name}_generation_galaxy(use_2d_mesh: bool):
    prompts = [
        "{input}",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": "{model}",
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "use_2d_mesh": use_2d_mesh,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")

    check_host_memory("{model}")
```

---

## Steps

1. Parse `model`, `device`, and optional `input` from the user's message. Ask if `model` or `device` are missing.

2. Derive `{snake_name}` from the model ID:
   - Take the part after the last `/` (e.g. `Llama-3.2-3B` from `meta-llama/Llama-3.2-3B`)
   - Lowercase and replace `-` and `.` with `_` (e.g. `llama_3_2_3b`)

3. Decide action based on device and whether `input` was provided:

   **single / n150 / p150 — always new function:**
   - Read `tests/integrations/vllm_plugin/generative/test_llama3_3b_generation.py`
   - Append the single-device standalone function at the end, substituting `{model}`, `{snake_name}`, `{input}` (default: `"I like taking walks in the"`)

   **n300-llmbox / galaxy WITH `input` — new standalone function:**
   - Read `tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py`
   - Append the appropriate standalone function template at the end, substituting `{model}`, `{snake_name}`, `{input}`

   **n300-llmbox / galaxy WITHOUT `input` — add to parametrize list:**
   - Read `tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py`
   - Find the `test_tensor_parallel_generation_llmbox_large` parametrize block:
     ```python
     @pytest.mark.parametrize(
         ["model_name", "enable_const_eval", "experimental_weight_dtype", "use_2d_mesh"],
         [
             pytest.param(...),
             ...last entry...
         ],
     )
     ```
   - Insert a new line after the last `pytest.param(...)` entry, before the closing `],`:
     ```python
             pytest.param("{model}", False, "", "True"),
     ```

4. Write the edit. Show the user exactly what was added or changed.

5. Run the test automatically using Bash. Do not suggest the command — execute it directly:
   - **single / n150 / p150**: `pytest -svv tests/integrations/vllm_plugin/generative/test_llama3_3b_generation.py::test_{snake_name}_generation -m single_device`
   - **n300-llmbox (with input)**: `pytest -svv tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py::test_{snake_name}_generation_n300_llmbox -m llmbox`
   - **galaxy (with input)**: `pytest -svv tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py::test_{snake_name}_generation_galaxy -m galaxy`
   - **n300-llmbox / galaxy (no input)**: `pytest -svv tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py::test_tensor_parallel_generation_llmbox_large -m llmbox`

   After running, report the full pytest output and whether the test passed or failed.
