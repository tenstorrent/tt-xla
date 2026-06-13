---
name: llm-model-bringup
description: Automate LLM bringup end-to-end — generate a `loader.py` for a new LLM inside a `/tmp/tt-forge-models` clone of `tenstorrent/tt-forge-models`, raise a PR against that repo, and trigger the corresponding tt-xla test on CI via the `automate-ci-runs` skill. Never edits the user's working tree. Use when the user wants to bring up a new LLM (e.g. "bring up Mistral 7B", "add a new LLM model", "stand up <HF model id>").
---

# LLM Model Bringup

## Step 1 — Collect bringup parameters from the user

Before doing anything else, ask the user for the three required parameters **in a single plain-text message** — a numbered list, no `AskUserQuestion` tool. The user replies with all three values in one message.

Print the message in this shape (adapt the wording, keep the structure):

> To bring up a new LLM, I need the following from you:
>
> 1. **`model_id`** — the HuggingFace model id of the LLM to bring up (e.g. `mistralai/Mistral-7B-v0.1`, `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct`). This is the value passed to `AutoModelForCausalLM.from_pretrained(...)`.
> 2. **`input_type`** — the modality the model accepts. One of:
>    - `text` — text-only LLM (causal LM); inputs are `input_ids` + `attention_mask`.
>    - `text+image` — multimodal LLM (vision-language); inputs include image tensors alongside text.
> 3. **`inference_type`** — the parallelism mode the test will run in. One of:
>    - `single` — single-device inference (`single_device` inference, runs on n150/n300/p150).
>    - `parallel` — tensor-parallel inference (`tensor_parallel` inference, runs on multi-device runners like n300-llmbox, galaxy-wh-6u).
> 4. **`family_folder`** — the folder name to use under `/tmp/tt-forge-models/` (the working clone — see Step 1.6) for this model. Snake_case, lowercase, includes the family version when applicable. Examples: `mistral`, `llama_3_2`, `qwen_2_5`, `qwen_3`, `command_r`, `deepseek_v4`, `gemma_3`. If the folder already exists upstream, it will be reused as-is (no overwrite).
>
> The four values above are **required**. The following are **optional** — if the user does not provide them, fall back to the defaults shown:
>
> 5. **`tokenizer_class`** *(optional)* — the HuggingFace tokenizer/processor class used in the generated loader. Defaults are picked by `input_type`:
>    - `input_type=text` → `AutoTokenizer`
>    - `input_type=text+image` → `AutoProcessor`
>
>    Override only if the model requires a specific class (e.g. `LlamaTokenizer`, `Qwen2Tokenizer`, `LlavaProcessor`).
> 6. **`model_class`** *(optional)* — the HuggingFace model class used in the generated loader. Defaults are picked by `input_type`:
>    - `input_type=text` → `AutoModelForCausalLM`
>    - `input_type=text+image` → `AutoModelForImageTextToText`
>
>    Override only if the model requires a specific class (e.g. `LlavaForConditionalGeneration`, `Qwen2VLForConditionalGeneration`).
> 7. **`trust_remote_code`** *(optional)* — whether to pass `trust_remote_code=True` to `from_pretrained(...)` calls. Default: `False`. Set to `True` only for models that ship custom modeling code on the Hub.
> 8. **`enable_weight_bfp8_conversion`** *(optional)* — whether to ask the tt-xla test runner to convert model weights to BFP8 at load time. Default: `False`. Set to `True` only when you specifically want the BFP8 path exercised for this bringup. The flag is consumed by Step 7 to decide whether to add an extra `enable_weight_bfp8_conversion: true` line under the test entry in the test-config YAML — it does **not** affect the generated `loader.py` itself.
> 9. **`arch`** *(optional)* — the target Tenstorrent runner/hardware label the CI dispatch will target in later steps. Defaults are picked by `inference_type`:
>    - `inference_type=single` → `n150`
>    - `inference_type=parallel` → `n300-llmbox`
>
>    Override only if you want the test to land on a different runner (e.g. `n300`, `p150`, `galaxy-wh-6u`). Whatever the user types is stored verbatim and overrides the default.
>
> Please reply with the required values (1–4) so I can continue. Mention 5–9 only if you want to override the defaults.

Then wait for the user's reply. Parse out the four required values from whatever format they send (comma-separated, line-by-line, labeled, etc. — be flexible). Validate:

- `model_id` is non-empty and looks like a HuggingFace id (`<org>/<name>` form). If it doesn't match, re-ask just this item.
- `input_type` is exactly one of `text` or `text+image`. If not, re-ask just this item.
- `inference_type` is exactly one of `single` or `parallel`. If not, re-ask just this item.
- `family_folder` is non-empty, lowercase, snake_case (letters/digits/underscores only). If not, re-ask just this item.

If any of the four required values are missing or invalid, re-ask only the missing/invalid item — do not re-prompt for values the user already gave correctly. Store the final answers as `model_id`, `input_type`, `inference_type`, `family_folder`, `tokenizer_class`, `model_class`, `trust_remote_code`, `enable_weight_bfp8_conversion`, and `arch` for use in later steps. For optional values the user did not supply, use the defaults defined above (do **not** ask again for optional values — silently apply defaults).

For `arch` specifically: the default depends on `inference_type`, so the resolution order is (1) explicit user override → (2) `n150` if `inference_type=single`, `n300-llmbox` if `inference_type=parallel`. Compute `arch`'s default **after** `inference_type` has been validated.

### Step 1.5 — Print the collected values for user confirmation

After all nine values are resolved (required + optional defaults filled in), print them back to the user **once** in a compact block so they can re-check before the skill proceeds. Use this exact shape (substitute the resolved values):

> Please confirm the following before I continue:
>
> - **model_id**: `<model_id>`
> - **input_type**: `<input_type>`
> - **inference_type**: `<inference_type>`
> - **family_folder**: `<family_folder>`
> - **tokenizer_class**: `<tokenizer_class>`  *(default)* or *(user-provided)*
> - **model_class**: `<model_class>`  *(default)* or *(user-provided)*
> - **trust_remote_code**: `<trust_remote_code>`  *(default)* or *(user-provided)*
> - **enable_weight_bfp8_conversion**: `<enable_weight_bfp8_conversion>`  *(default)* or *(user-provided)*
> - **arch**: `<arch>`  *(default)* or *(user-provided)*
>
> Reply `ok` / `yes` / `proceed` to continue, or tell me which value to change.

Wait for the user's response.

- If the user confirms (`ok`, `yes`, `proceed`, `looks good`, etc.) → continue to Step 2.
- If the user asks to change one or more values → update only those values, re-print the full block, and wait again. Do not proceed until the user confirms.

Mark each value as *(default)* if it came from the fallback defaults and *(user-provided)* if the user supplied it explicitly — this makes it obvious which fields the user has not yet reviewed.

### Step 1.6 — Set up working clones in `/tmp`

**Never edit the user's working copy (e.g. `/home/tt-xla/...`).** All file edits performed by this skill happen inside `/tmp/` clones of the upstream repos. This keeps the user's local checkout untouched and ensures we always work against a fresh tree from `main`.

For each repo the skill touches:

- If `/tmp/<repo>/` **does not exist** → `git clone <upstream-url> /tmp/<repo>`.
- If `/tmp/<repo>/` **already exists** → `git -C /tmp/<repo> pull origin` (every invocation, no matter how recent the clone is — we want the latest `main`).

Repos this skill uses:

| Purpose                                                       | Local path in `/tmp`     | Upstream URL                                          |
| ------------------------------------------------------------- | ------------------------ | ----------------------------------------------------- |
| Add the new `loader.py` (Steps 2–3, and PR target)            | `/tmp/tt-forge-models`   | `https://github.com/tenstorrent/tt-forge-models.git`  |
| Edit tt-xla test configs / dispatch CI (later steps)          | `/tmp/tt-xla`            | `https://github.com/tenstorrent/tt-xla.git`           |

For Steps 2–5 only the **`tt-forge-models`** clone is required. Set it up now:

```bash
if [ -d /tmp/tt-forge-models/.git ]; then
  git -C /tmp/tt-forge-models pull origin
else
  git clone https://github.com/tenstorrent/tt-forge-models.git /tmp/tt-forge-models
fi
```

The `tt-xla` clone is lazily created later when a step explicitly needs it (e.g. test-config edits, CI dispatch). Do not clone it preemptively in this step.

**All path references from here onward** (Steps 2, 3, 5, …) refer to `/tmp/tt-forge-models/<family_folder>/...`, **not** `third_party/tt_forge_models/<family_folder>/...` inside the user's tree.

## Step 2 — Create the model folder structure in `tt_forge_models`

Create the directory tree and `__init__.py` for the new model inside the `/tmp/tt-forge-models/` clone set up in Step 1.6. The `loader.py` content itself is filled in by later steps — this step only lays down the skeleton.

### 2a — Use the user-provided `family_folder`

The family folder name is collected directly from the user in Step 1 (as the required `family_folder` value). **Do not derive it from `model_id`** — use the value the user gave verbatim.

Behavior:

- If `/tmp/tt-forge-models/<family_folder>/` **already exists** (i.e. upstream already has this family), reuse it as-is. Do **not** overwrite the existing root folder, and do **not** create any new files at that level — only the task subfolder in 2b–2c.
- If it does not exist, it will be created as part of 2c (as parents of the `pytorch/` subfolder).

Reference (for the human guiding the skill — these are real examples the user may type, mirroring the existing repo convention):

| `model_id`                              | `family_folder` the user typically supplies |
| --------------------------------------- | -------------------------------------------- |
| `mistralai/Mistral-7B-v0.1`             | `mistral`                                    |
| `meta-llama/Llama-3.2-3B-Instruct`      | `llama_3_2`                                  |
| `Qwen/Qwen2.5-7B-Instruct`              | `qwen_2_5`                                   |
| `Qwen/Qwen3-32B`                        | `qwen_3`                                     |
| `deepseek-ai/DeepSeek-V4`               | `deepseek_v4`                                |
| `google/gemma-3-27b`                    | `gemma_3`                                    |
| `CohereLabs/c4ai-command-r-v01`         | `command_r`                                  |
| `CohereForAI/c4ai-command-a-reasoning`  | `command_c4ai`                               |

Note: this table is informational only. The skill does **not** apply any rules to compute these — they are whatever the user typed.

### 2b — Determine the task subfolder from `input_type`

| `input_type`  | Task subfolder       | Docstring phrase                                  |
| ------------- | -------------------- | ------------------------------------------------- |
| `text`        | `causal_lm`          | `causal language modeling`                        |
| `text+image`  | `multimodal`         | `multimodal (image + text) language modeling`     |

Note: the existing repo also uses task subfolder names like `image_to_text` for some vision-language models (e.g. `qwen_3_vl/image_to_text/`). For this skill we standardize on `multimodal` for `text+image`. If the user explicitly requests a different subfolder name for a given bringup, accept the override and use what they specified.

### 2c — Create the directory and `loader.py` placeholder

Create (inside the `/tmp/tt-forge-models/` clone from Step 1.6):

```
/tmp/tt-forge-models/<family_folder>/<task_subfolder>/pytorch/__init__.py
/tmp/tt-forge-models/<family_folder>/<task_subfolder>/pytorch/loader.py   (empty placeholder — filled in by later steps)
```

First `mkdir -p` the `pytorch/` directory (so parent levels are created if missing), then use the `Write` tool for both files. The `__init__.py` and `loader.py` live together inside the `pytorch/` subfolder; do **not** create extra `__init__.py` files at the `<family_folder>/` or `<task_subfolder>/` levels — those resolve as implicit namespace packages, matching the existing repo convention.

### 2d — Write the `__init__.py` content

The `__init__.py` placed in `/tmp/tt-forge-models/<family_folder>/<task_subfolder>/pytorch/__init__.py` must follow this exact template:

```python
# SPDX-FileCopyrightText: (c) <YEAR> Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
<MODEL DISPLAY NAME> <DOCSTRING PHRASE> implementation for Tenstorrent projects.
"""
# Import from the loader module
from .loader import ModelLoader
```

Substitutions:

- `<YEAR>` → the **current** year as a 4-digit integer (use today's date — do not hardcode `2025` or `2026`; read the date at run time).
- `<MODEL DISPLAY NAME>` → human-readable name derived from `model_id`. Examples:
  - `mistralai/Mistral-7B-v0.1` → `Mistral`
  - `meta-llama/Llama-3.2-3B-Instruct` → `Llama 3.2 3B Instruct`
  - `CohereForAI/c4ai-command-a-reasoning` → `Command A Reasoning`
  - `Qwen/Qwen2.5-7B-Instruct` → `Qwen 2.5 7B Instruct`
- `<DOCSTRING PHRASE>` → from the table in 2b (e.g. `causal language modeling` for `text`).

**Important:** the docstring must read naturally for the chosen model — show the assembled `__init__.py` content to the user for one quick confirmation before writing it, in case the derived display name needs a tweak.

### 2e — End-of-step sanity check

After writing the two files, list them back to the user in a short confirmation block:

> Created:
> - `/tmp/tt-forge-models/<family_folder>/<task_subfolder>/pytorch/__init__.py`
> - `/tmp/tt-forge-models/<family_folder>/<task_subfolder>/pytorch/loader.py` (empty placeholder — to be filled in next)

Then continue to Step 3.

## Step 3 — Handle the input

This step fills in the `loader.py` placeholder created in Step 2 with the body appropriate for the confirmed `input_type` from Step 1.5. Select the branch directly from that value — no further prompting:

- If `input_type == "text"` → use branch 3a.
- If `input_type == "text+image"` → use branch 3b.

### 3a — `input_type == "text"` branch

Write the following content to `/tmp/tt-forge-models/<family_folder>/<task_subfolder>/pytorch/loader.py`, applying the substitutions listed below.

Template:

```python
# SPDX-FileCopyrightText: (c) <YEAR> Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
<MODEL DISPLAY NAME> causal LM model loader implementation.
"""

import torch
from transformers import <TOKENIZER_CLASS>, <MODEL_CLASS>
from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available <MODEL DISPLAY NAME> model variants for causal language modeling."""

    <VARIANT_UPPER> = "<VARIANT_LOWER>"


class ModelLoader(ForgeModel):
    """<MODEL DISPLAY NAME> model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.<VARIANT_UPPER>: LLMModelConfig(
            pretrained_model_name="<MODEL_ID>",
            max_length=256,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.<VARIANT_UPPER>

    sample_text = "How many r's are there in strawberry?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="<MODEL_NAME_SNAKE>",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self):
        """Load tokenizer for the current variant.
        Returns:
            The loaded tokenizer instance
        """
        self.tokenizer = <TOKENIZER_CLASS>.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the <MODEL DISPLAY NAME> model for this instance's variant.

        Args:
                           If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The model for causal language modeling.
        """
        pretrained_model_name = self._variant_config.pretrained_model_name

        if self.tokenizer is None:
            self._load_tokenizer()

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = <MODEL_CLASS>.from_pretrained(
            pretrained_model_name, **model_kwargs
        )

        model.eval()
        self.config = model.config
        self.model = model
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the <MODEL DISPLAY NAME> model.

        Args:
            batch_size: Batch size for the inputs.

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self._load_tokenizer()

        max_length = self._variant_config.max_length
        conversation = [{"role": "user", "content": self.sample_text}]
        prompt = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)
        return inputs
```

Substitutions (apply all of these consistently — never leave a `<...>` placeholder in the written file):

- `<YEAR>` → the **current** 4-digit year (read today's date at run time; do not hardcode `2025` or `2026`).
- `<MODEL DISPLAY NAME>` → human-readable name derived from `model_id`, same value used in Step 2d (e.g. `Command A Reasoning`, `Mistral`, `Llama 3.2 3B Instruct`). Use this verbatim in:
  - the module docstring (line 5 of the template),
  - the `ModelVariant` class docstring,
  - the `ModelLoader` class docstring,
  - the `load_model` docstring,
  - the `load_inputs` docstring.
- `<MODEL_NAME_SNAKE>` → `<MODEL DISPLAY NAME>` with spaces replaced by underscores, casing preserved (e.g. `Command A Reasoning` → `Command_A_Reasoning`, `Llama 3.2 3B Instruct` → `Llama_3.2_3B_Instruct`). Used as the `model="..."` field inside `ModelInfo(...)`.
- `<MODEL_ID>` → the full HuggingFace id from Step 1, verbatim (e.g. `CohereLabs/command-a-reasoning-08-2025`). Used as the `pretrained_model_name="..."` value inside `LLMModelConfig(...)`.
- `<VARIANT_LOWER>` → the slug portion **after the `/`** in `model_id`, kept lowercase with hyphens between words (e.g. `command-a-reasoning-08-2025` from `CohereLabs/command-a-reasoning-08-2025`).
- `<VARIANT_UPPER>` → `<VARIANT_LOWER>` with hyphens converted to underscores and uppercased (e.g. `COMMAND_A_REASONING_08_2025`). Used as the `ModelVariant` enum member name and in `_VARIANTS` / `DEFAULT_VARIANT`.
- `<TOKENIZER_CLASS>` → the resolved `tokenizer_class` from Step 1 (default `AutoTokenizer` for `text`). Must match in **both** the `from transformers import ...` line and the `<TOKENIZER_CLASS>.from_pretrained(...)` call.
- `<MODEL_CLASS>` → the resolved `model_class` from Step 1 (default `AutoModelForCausalLM` for `text`). Must match in **both** the `from transformers import ...` line and the `<MODEL_CLASS>.from_pretrained(...)` call.

Additional rules:

- If `trust_remote_code == True` (from Step 1), pass `trust_remote_code=True` to **both** `from_pretrained(...)` calls — add it directly to the tokenizer call in `_load_tokenizer`, and add it to `model_kwargs` before the `<MODEL_CLASS>.from_pretrained(...)` call in `load_model`.
- Do **not** alter the `sample_text` constant, the `LLMModelConfig.max_length` value (`256`), or the `ModelGroup.RED` / `ModelTask.NLP_CAUSAL_LM` / `ModelSource.HUGGING_FACE` / `Framework.TORCH` fields — these are fixed for the text branch.
- After writing, scan the file once for any remaining `<` / `>` placeholder pairs. If any are left, the substitution table above was applied incompletely — fix before continuing.

### 3b — `input_type == "text+image"` branch

*(To be defined — the user will provide the contents of this branch.)*

## Step 4 — Filter on `inference_type`

Inspect the confirmed `inference_type` from Step 1.5:

- If `inference_type == "single"` → **skip this step entirely** and continue to the next step.
- If `inference_type == "parallel"` → run the parallel-specific logic below.

*(Parallel-specific logic to be defined — the user will provide the contents.)*

This step is responsible for emitting the parallel-only loader pieces (e.g. `get_mesh_config`, `load_shard_spec`) when the test will run tensor-parallel. For single-device runs, none of this is needed and the step is a no-op.

## Step 5 — Assemble the pytest `test_name` for the new model

This step builds the pytest selector string that targets the new loader inside `tests/runner/test_models.py::test_all_models_torch[...]`. No files are written in this step — its output is a single string that is **printed back to the user** so it can be used in Step 6+ (CI dispatch, manual reproduction, etc.).

### 5a — Build the `test_name`

The `test_name` has this exact shape:

```
<family_folder>/<task_subfolder>/pytorch-<variant_slug>-<inference_type_full>-inference
```

where:

- `<family_folder>` — the value from Step 1 (e.g. `coherlabs`, `gpt_neo`, `mistral`).
- `<task_subfolder>` — the value from Step 2b (e.g. `causal_lm` for `text`, `multimodal` for `text+image`, or an explicit override).
- `pytorch` — the framework subfolder under which `loader.py` lives in `/tmp/tt-forge-models/` (from Step 2c). It is **glued to the variant slug with a hyphen, not a slash**, because the test-id generator in `tests/runner/utils/dynamic_loader.py::generate_test_id` (over in tt-xla) joins the loader's directory path with the variant via `-`: `f"{model_path}-{variant_name}"`. So `coherlabs/causal_lm/pytorch/loader.py` + variant `c4ai-command-r-v01` produces `coherlabs/causal_lm/pytorch-c4ai-command-r-v01`, not `coherlabs/causal_lm/pytorch/c4ai-command-r-v01`.
- `<variant_slug>` — the **lowercase-hyphen variant slug** from the loader, i.e. the right-hand side of the `ModelVariant` enum (`<VARIANT_LOWER>` from Step 3a — e.g. `c4ai-command-r-v01`, `2_7B`). Do **not** use `<VARIANT_UPPER>` and do **not** use the full `<MODEL_ID>` here.
- `<inference_type_full>` — the long form of `inference_type` from Step 1.5:
  - `inference_type == "single"` → `single_device`
  - `inference_type == "parallel"` → `tensor_parallel`

Worked example for the reference target `gpt_neo/sequence_classification/pytorch-2_7B-single_device-inference`:

| Field                  | Value                          |
| ---------------------- | ------------------------------ |
| `family_folder`        | `gpt_neo`                      |
| `task_subfolder`       | `sequence_classification`      |
| Framework subfolder    | `pytorch`                      |
| `<variant_slug>`       | `2_7B`                         |
| `<inference_type_full>`| `single_device`                |

### 5b — Wrap with the pytest selector

Combine the `test_name` into the full pytest invocation target:

```
tests/runner/test_models.py::test_all_models_torch[<test_name>]
```

### 5c — Print the result to the user

Emit one short block back to the user, in this exact shape (substitute the assembled values):

> **Generated test selector**
>
> - `test_name`: `<test_name>`
> - Full pytest target: `tests/runner/test_models.py::test_all_models_torch[<test_name>]`

Do not write this string to any file in this step — Step 6 onwards (CI dispatch via `automate-ci-runs`, etc.) will consume it.

## Step 6 — Raise the PR in `tenstorrent/tt-forge-models`

This step takes the new `loader.py` + `__init__.py` written under `/tmp/tt-forge-models/<family_folder>/<task_subfolder>/pytorch/` and ships them upstream as a pull request against `tenstorrent/tt-forge-models`. All git operations happen inside the `/tmp/tt-forge-models/` clone set up in Step 1.6 — never in the user's `/home/tt-xla/` tree.

### 6a — Compute branch name, commit message, and PR title

All three use the **variant slug** from Step 3a (the lowercase-hyphen `<VARIANT_LOWER>`, i.e. the right-hand side of the `ModelVariant` enum — e.g. `c4ai-command-r-v01`, **not** the full HuggingFace id with the `/`):

| Field           | Format                                | Example                                |
| --------------- | ------------------------------------- | -------------------------------------- |
| Branch name     | `bringup_<variant_slug>`              | `bringup_c4ai-command-r-v01`           |
| Commit message  | `Bringup the <variant_slug>`          | `Bringup the c4ai-command-r-v01`       |
| PR title        | same as commit message                | `Bringup the c4ai-command-r-v01`       |

Why the slug and not the full `model_id`? Git branch names can't contain `/` cleanly (it implies a namespace), and slugs are the canonical short identifier used elsewhere in this skill (variant enum value, test_name).

### 6b — Confirm the push plan with the user

Before any destructive / externally-visible action (`git push`, `gh pr create`), print a compact plan block back to the user and **wait for explicit confirmation**:

> About to push to `tenstorrent/tt-forge-models` and open a **draft** PR:
>
> - **Branch**: `<branch_name>`
> - **Commit message**: `<commit_msg>`
> - **PR title**: `<pr_title>`
> - **PR base**: `main`
> - **PR state**: draft (you flip to "Ready for review" yourself once verified)
> - **Files added**:
>   - `<family_folder>/<task_subfolder>/pytorch/__init__.py`
>   - `<family_folder>/<task_subfolder>/pytorch/loader.py`
>
> Reply `ok` to proceed, or tell me to adjust any field.

Do **not** push or create the PR until the user confirms.

### 6c — Create branch, stage, and commit

Inside `/tmp/tt-forge-models/`:

```bash
cd /tmp/tt-forge-models
git checkout -b <branch_name>
git add <family_folder>/<task_subfolder>/pytorch/__init__.py \
        <family_folder>/<task_subfolder>/pytorch/loader.py
git commit -m "<commit_msg>"
```

- Always start from a fresh `main` (Step 1.6 already ran `git pull origin`).
- Stage **only the two new files** with explicit paths — do **not** `git add -A` or `git add .`, which could pick up unrelated stray files in `/tmp/tt-forge-models/`.
- If the branch already exists locally from a previous attempt, delete it (`git branch -D <branch_name>`) before re-creating — but only after confirming with the user.

### 6d — Push and open the PR (as a **draft**)

PRs raised by this skill are always opened as **draft PRs** (`--draft`). The author flips them to "Ready for review" themselves once they have verified the model loads, the test passes locally / on a manual CI run, and any downstream tt-xla wiring is in place. The skill should not mark anything as ready.

```bash
git push -u origin <branch_name>

gh pr create \
  --repo tenstorrent/tt-forge-models \
  --base main \
  --head <branch_name> \
  --draft \
  --title "<pr_title>" \
  --body "$(cat <<'EOF'
## Summary

Bring up <MODEL DISPLAY NAME> (`<MODEL_ID>`) for the `<inference_type_full>` inference path on Tenstorrent hardware.

## Changes

- Add `<family_folder>/<task_subfolder>/pytorch/__init__.py` — re-exports `ModelLoader`.
- Add `<family_folder>/<task_subfolder>/pytorch/loader.py` — `<MODEL DISPLAY NAME>` loader with variant `<VARIANT_UPPER> = "<VARIANT_LOWER>"`, using `<TOKENIZER_CLASS>` + `<MODEL_CLASS>` from `transformers`.

## Test target (in tt-xla)

`tests/runner/test_models.py::test_all_models_torch[<test_name>]`

Generated by the `llm-model-bringup` skill.
EOF
)"
```

If the remote rejects the push because the user lacks write access to `tenstorrent/tt-forge-models`, fall back to a fork-and-push flow: tell the user the push was rejected, and ask whether to push to their fork instead (do not silently create a fork).

### 6e — Print the PR URL

`gh pr create` prints the PR URL on stdout. Capture it and echo it back to the user in a short confirmation block:

> ✅ PR opened: `<PR_URL>`
>
> - Branch: `<branch_name>` → `tenstorrent/tt-forge-models@main`
> - Title: `<pr_title>`
>
> The PR description has been populated with the summary, file list, and tt-xla test target.

If, after creation, the user asks to tweak the description, use `gh pr edit <PR_URL> --body "..."` to update it in place (do not close-and-reopen).

## Step 7 — Wire the test into tt-xla and raise the test branch

This step crosses over into the **tt-xla** repo (cloned for the first time here, lazily, per the table in Step 1.6). It (a) appends the new `test_name` to the right `test_config` YAML, (b) bumps the `third_party/tt_forge_models/` submodule to the bringup branch from Step 6 so CI picks up the new loader, then (c) pushes a tt-xla branch — **no PR** is opened here, just the branch.

### 7a — Set up the `/tmp/tt-xla` clone (with submodule init)

Same clone-or-pull pattern as Step 1.6, but for tt-xla. **Whichever branch fires (clone OR pull), `git submodule update --init --recursive` MUST run before anything else in Step 7.** Without it, `third_party/tt_forge_models/` is empty / out-of-sync and Step 7d's submodule checkout cannot succeed:

```bash
if [ -d /tmp/tt-xla/.git ]; then
  git -C /tmp/tt-xla checkout main
  git -C /tmp/tt-xla pull origin
  git -C /tmp/tt-xla submodule update --init --recursive   # REQUIRED every invocation
else
  git clone https://github.com/tenstorrent/tt-xla.git /tmp/tt-xla
  git -C /tmp/tt-xla submodule update --init --recursive   # REQUIRED every invocation
fi
```

**Run `git submodule update --init --recursive` on EVERY entry into Step 7**, even when the clone already exists and pulled cleanly — submodule pointers can drift between `main` updates, and we need `third_party/tt_forge_models/` to be a real, populated git working tree before Step 7d cd's into it and switches branches.

Why this matters: `tests/runner/utils/dynamic_loader.py` scans `third_party/tt_forge_models/` at parametrize time, and Step 7d switches that submodule to the `bringup_<variant_slug>` branch — both operations require the submodule to be initialized first.

### 7b — Pick the right test-config YAML based on `inference_type`

| `inference_type` | Path inside `/tmp/tt-xla`                                                   |
| ---------------- | --------------------------------------------------------------------------- |
| `single`         | `tests/runner/test_config/torch/test_config_inference_single_device.yaml`   |
| `parallel`       | `tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml` |

### 7c — Append the new test entry to the YAML

Append at the **bottom** of the chosen YAML (use the `Read` tool first to inspect existing indentation; match it exactly — the existing entries use 2 spaces for the test key and 4 spaces for child keys like `status:`):

- If `enable_weight_bfp8_conversion == False` (the default), write **two lines**:

  ```yaml
    <test_name>:
      status: EXPECTED_PASSING
  ```

- If `enable_weight_bfp8_conversion == True`, write **three lines** — the same `status:` line plus an extra `enable_weight_bfp8_conversion: true` sibling key, indented identically:

  ```yaml
    <test_name>:
      status: EXPECTED_PASSING
      enable_weight_bfp8_conversion: true
  ```

`<test_name>` is the exact string from Step 5a (e.g. `coherlabs/causal_lm/pytorch-c4ai-command-r-v01-single_device-inference`). Do **not** add any wrapping section header or comment — just the entry, indented to match siblings. The BFP8 flag is the **only** YAML difference this skill emits between the two modes; everything else (path selection, branch, push) is identical.

### 7d — Pin the `tt_forge_models` submodule at the bringup branch

**Prerequisite:** Step 7a's `git submodule update --init --recursive` MUST have run in this invocation. If you're unsure (e.g. resuming mid-skill), re-run it before the commands below. Without the submodule init, `cd /tmp/tt-xla/third_party/tt_forge_models` lands in an empty or un-initialized directory and the `git fetch` / `git checkout` will fail.

The tt-xla submodule at `third_party/tt_forge_models/` must point at the branch we pushed in Step 6 (`bringup_<variant_slug>`) so CI picks up the new loader:

```bash
git -C /tmp/tt-xla submodule update --init --recursive   # safety net — idempotent re-run
cd /tmp/tt-xla/third_party/tt_forge_models
git fetch origin <bringup_branch>
git checkout <bringup_branch>
cd /tmp/tt-xla
```

After the checkout, `git status` from `/tmp/tt-xla` will show `third_party/tt_forge_models` as a modified submodule pointer — that change is staged as part of the tt-xla branch commit in 7f.

### 7e — Confirm the push plan with the user

Before any destructive / externally-visible action, print a plan block and **wait for explicit confirmation**:

> About to push a tt-xla branch (no PR):
>
> - **Branch**: `ci_testing_<variant_slug>`
> - **Commit message**: `ci testing for <variant_slug>`
> - **Files changed**:
>   - `tests/runner/test_config/torch/<yaml_path>` (added `<test_name>` entry, BFP8 = `<enable_weight_bfp8_conversion>`)
>   - `third_party/tt_forge_models` submodule → `<bringup_branch>`
> - **No PR** is created — this is branch-only, for manual CI dispatch.
>
> Reply `ok` to proceed.

### 7f — Branch, stage, commit, push (no PR)

```bash
cd /tmp/tt-xla
git checkout -b ci_testing_<variant_slug>
git add tests/runner/test_config/torch/<yaml_path> third_party/tt_forge_models
git commit -m "ci testing for <variant_slug>"
git push -u origin ci_testing_<variant_slug>
```

- Stage **only** the two specific paths (YAML + submodule) — do not `git add -A`.
- **Do not run `gh pr create`.** The user explicitly wants the branch raised on its own; the PR is opened later (or never — the branch is often used only for ad-hoc CI runs).

### 7g — Confirm to the user

Print a short confirmation block with the pushed branch URL:

> ✅ tt-xla branch pushed: `ci_testing_<variant_slug>` → `tenstorrent/tt-xla`
>
> - YAML updated: `<yaml_path>` (added `<test_name>: status: EXPECTED_PASSING`)
> - Submodule pinned: `third_party/tt_forge_models` → `<bringup_branch>`
> - No PR created — branch-only, ready for manual CI dispatch (see `automate-ci-runs` skill).

## Step 8 — Trigger the CI run via the `automate-ci-runs` skill

This is the **final step** of the bringup workflow. Rather than reimplementing the workflow-dispatch logic, hand off to the existing `automate-ci-runs` skill (which dispatches the `Run Test Single` workflow / `.github/workflows/manual-test-single.yml` on `tenstorrent/tt-xla`). This skill only needs to **pre-populate the three values that skill collects in its own Step 1** so the user doesn't have to re-enter them.

### 8a — Compute the three handoff values

All three derive directly from earlier steps — no new user input required:

| Field                   | Value                                                                                                                       | Source                          |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| `branch_name`           | `ci_testing_<variant_slug>` (e.g. `ci_testing_c4ai-command-r-v01`)                                                          | Step 7f (the pushed branch)     |
| `full_command_to_test`  | `tests/runner/test_models.py::test_all_models_torch[<test_name>]` (e.g. `tests/runner/test_models.py::test_all_models_torch[coherlabs/causal_lm/pytorch-c4ai-command-r-v01-single_device-inference]`) | Step 5b (the full pytest target) |
| `machine_name`          | `<arch>` (e.g. `n150` for `single` default, `n300-llmbox` for `parallel` default)                                           | Step 1 #9 (user-provided or `inference_type`-derived default) |

### 8b — Confirm the handoff with the user

Before invoking the other skill, print a compact block so the user sees exactly what will be dispatched:

> About to hand off to `automate-ci-runs` to dispatch the CI workflow with:
>
> - **branch_name**: `<branch_name>`
> - **full_command_to_test**: `<full_command_to_test>`
> - **machine_name**: `<machine_name>`
>
> Reply `ok` to dispatch, or tell me which value to change.

Do **not** dispatch until the user confirms.

### 8c — Invoke the `automate-ci-runs` skill

Invoke the `automate-ci-runs` skill (via the `Skill` tool, `skill: automate-ci-runs`). Pass the three pre-computed values so its Step 1 can skip re-collection — supply them in the invocation message, e.g.:

> Pre-filled values for `automate-ci-runs`:
> - `branch_name`: `<branch_name>`
> - `full_command_to_test`: `<full_command_to_test>`
> - `machine_name`: `<machine_name>`
>
> Please dispatch the `Run Test Single` workflow on `tenstorrent/tt-xla` with these values.

`automate-ci-runs` owns everything from there (validation, `gh workflow run`, surfacing the run URL). When it returns, relay the run URL it produced back to the user.

### 8d — End of bringup

After `automate-ci-runs` reports the dispatched run URL, the bringup is complete. Summarize the full delivery to the user:

> 🎉 Bringup complete for `<MODEL DISPLAY NAME>` (`<MODEL_ID>`):
>
> - tt-forge-models draft PR: `<PR_URL_from_Step_6>`
> - tt-xla branch: `ci_testing_<variant_slug>` (no PR)
> - CI workflow dispatched: `<run_URL_from_automate-ci-runs>`
>
> Once the CI run finishes, flip the draft PR to "Ready for review" and (optionally) open a tt-xla PR for the `ci_testing_<variant_slug>` branch if the test should be merged.

