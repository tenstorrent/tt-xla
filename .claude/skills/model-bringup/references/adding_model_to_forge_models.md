# Adding a Model to tt-forge-models

Guide for adding a new model to the `tt-forge-models` repository so it can be discovered by the tt-xla test runner.

## Overview

Models live in `third_party/tt_forge_models/` (a git submodule pointing to https://github.com/tenstorrent/tt-forge-models). Each model must implement a `ModelLoader` class in a `loader.py` file.

## Directory structure

```
tt-forge-models/
└── <model_name>/
    └── <optional_task>/         # e.g., causal_lm, question_answering, image_classification
        └── pytorch/             # or jax/
            ├── loader.py        # Required: ModelLoader implementation
            ├── requirements.txt # Optional: per-model pip dependencies
            └── requirements.nodeps.txt  # Optional: deps installed with --no-deps
```

Examples:
- `resnet/pytorch/loader.py` — simple CV model
- `llama/causal_lm/pytorch/loader.py` — LLM with task subdirectory
- `bert/question_answering/pytorch/loader.py` — NLP with task subdirectory

## ModelLoader interface

The `ModelLoader` class must inherit from `ForgeModel` (from `tt-forge-models/base.py`) and implement:

### Required methods

```python
from ...config import ModelConfig, ModelInfo, ModelGroup, ModelTask, ModelSource, Framework, StrEnum
from ...base import ForgeModel

class ModelVariant(StrEnum):
    """Available variants for this model."""
    VARIANT_A = "variant_a_name"
    VARIANT_B = "variant_b_name"

class ModelLoader(ForgeModel):
    # Dict mapping variant enum to config
    _VARIANTS = {
        ModelVariant.VARIANT_A: ModelConfig(pretrained_model_name="org/model-name"),
        # ...
    }
    DEFAULT_VARIANT = ModelVariant.VARIANT_A

    def __init__(self, variant=None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant=None) -> ModelInfo:
        """Return metadata for dashboard reporting."""
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="ModelName",
            variant=variant,
            group=ModelGroup.GENERALITY,  # or ModelGroup.RED for priority models
            task=ModelTask.NLP_TEXT_GEN,   # see ModelTask enum for options
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load and return the model instance (torch.nn.Module)."""
        model = ...  # Load from HuggingFace, torchvision, timm, etc.
        model.eval()
        return model

    def load_inputs(self, dtype_override=None, batch_size=1, **kwargs):
        """Load and return sample inputs for the model."""
        inputs = ...  # Create or load sample input tensors
        return inputs
```

### Optional methods (for multi-chip / sharding)

```python
    def load_shard_spec(self, model, inputs, mesh, parallelism, num_devices):
        """Return sharding specification for tensor parallel."""
        pass

    @classmethod
    def get_mesh_config(cls, num_devices):
        """Return mesh shape tuple for tensor parallel, e.g., (1, num_devices)."""
        return (1, num_devices)
```

## Key enums

### ModelGroup
- `ModelGroup.RED` — Priority customer models (tracked closely)
- `ModelGroup.GENERALITY` — General coverage models

### ModelTask
Common values: `CV_IMAGE_CLS`, `CV_OBJECT_DET`, `NLP_TEXT_GEN`, `NLP_QA`, `NLP_SEQ_CLS`, `NLP_TOKEN_CLS`, `NLP_MASKED_LM`, `AUDIO_SPEECH_REC`, `MULTIMODAL`

### ModelSource
- `ModelSource.HUGGING_FACE`, `ModelSource.TIMM`, `ModelSource.TORCHVISION`, `ModelSource.CUSTOM`

## Per-model dependencies

If the model needs additional pip packages beyond the base environment:
1. Create `requirements.txt` next to `loader.py`
2. Optionally create `requirements.nodeps.txt` for packages that need `--no-deps`
3. The test runner automatically installs/uninstalls these per-test

## Testing locally before uplifting

```bash
# From tt-forge-models directory, verify the loader works
cd third_party/tt_forge_models
python -c "
from <model_path>.pytorch.loader import ModelLoader
variants = ModelLoader.query_available_variants()
print('Variants:', variants)
info = ModelLoader.get_model_info()
print('Info:', info)
"
```

## Uplifting to tt-xla

After the model is merged in tt-forge-models:

```bash
# Update the submodule to include the new model
cd /path/to/tt-xla
git submodule update --remote third_party/tt_forge_models
git add third_party/tt_forge_models
git commit -m "Uplift tt-forge-models submodule to include <model_name>"
```

Then verify discovery:
```bash
source venv/activate
pytest -q --collect-only -k "<model_name>" tests/runner/test_models.py
```
