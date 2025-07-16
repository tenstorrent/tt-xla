# tt-forge-models

A shared repository of model implementations that can be used across TT-Forge frontend repositories.

## Overview

This repository contains model implementations that are used for testing and benchmarking across various TT-Forge frontend repositories. The goal is to have a single source of truth for model implementations rather than duplicating code across different repositories.

## Structure

The repository is organized with models as top-level directories, and frameworks as subdirectories. This structure allows supporting multiple implementations of the same model across different frameworks (PyTorch, TensorFlow, JAX, etc.).

```
├── base.py                    # Base interface for all model loaders
├── yolov4/                    # YOLOv4 model implementation
│   ├── __init__.py            # Package initialization with imports from pytorch subdirectory
│   └── pytorch/               # PyTorch-specific implementation
│       ├── __init__.py        # Package initialization
│       ├── loader.py          # Public interface - ModelLoader class
│       └── src/               # Model implementation details
│           ├── yolov4.py      # Main model implementation
│           ├── downsample1.py # Model components
│           └── ...
├── yolov3/
│   ├── __init__.py
│   └── pytorch/
│       ├── __init__.py
│       ├── loader.py
│       └── src/
├── bert/
│   ├── __init__.py
│   └── pytorch/
│       ├── __init__.py
│       └── loader.py
└── ...
```

## Usage

Each model in tt-forge-models follows the same standardized interface pattern. All loaders use the class name `ModelLoader` for consistency, making it easy to write generic code that works with any model.

```python
# Import specifically from the PyTorch implementation
from third_party.tt_forge_models.yolov4.pytorch import ModelLoader
import torch

# Load model with default settings (uses PyTorch default dtype, typically float32)
model = ModelLoader.load_model()

# Load sample inputs with default settings
inputs = ModelLoader.load_inputs()

# Use the model
outputs = model(inputs)

# For TT hardware, you can explicitly override to bfloat16 for model and inputs
bfp16_model = ModelLoader.load_model(dtype_override=torch.bfloat16)
bfp16_inputs = ModelLoader.load_inputs(dtype_override=torch.bfloat16)
outputs = bfp16_model(bfp16_inputs)
```

## Adding New Models

When adding a new model to the repository, follow these guidelines:

1. Create a new top-level directory for your model (e.g., `mynewmodel/`)
2. Create a framework-specific subdirectory (e.g., `mynewmodel/pytorch/`)
3. Create a `loader.py` file within the framework directory that implements the ForgeModel interface with a class named `ModelLoader`
4. Create a `src/` directory for your model implementation files (if needed)
5. Create `__init__.py` files in both the top-level and framework directories
6. Implement the following standard methods:
   - `load_model()` - Returns a model instance
   - `load_inputs()` - Returns sample inputs for the model
   - `_get_model_info()` - Returns metadata about the model

7. Use the standardized enum values from `config.py`:
   - `ModelSource` - Use `HUGGING_FACE` for HuggingFace models, `TORCHVISION` for TorchVision models, etc.
   - `ModelTask` - Use standardized task enums like `NLP_CAUSAL_LM`, `CV_IMAGE_CLS`, `AUDIO_ASR`, etc.
   - `ModelGroup` - Use `RED` for customer requested models, `GENERALITY` for others

8. Always use a dedicated `ModelVariant` enum class with a `DEFAULT_VARIANT` class attribute:
   ```python
   class ModelVariant(StrEnum):
       """Available model variants."""
       BASE = "base"
       LARGE = "large"

   # Dictionary of available model variants
   _VARIANTS = {
       ModelVariant.BASE: ModelConfig(...),
       ModelVariant.LARGE: ModelConfig(...),
   }

   # Default variant to use
   DEFAULT_VARIANT = ModelVariant.BASE
   ```

Example implementation of a model loader:

```python
# mymodel/pytorch/loader.py
import torch
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)
from ...base import ForgeModel
from .src.model import MyModel

class ModelVariant(StrEnum):
    """Available model variants."""
    BASE = "base"

class ModelLoader(ForgeModel):
    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="my_model_pretrained",
        )
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    # Shared configuration parameters as class variables
    param1 = "default_value"
    param2 = 42

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        return ModelInfo(
            model="my_model",
            variant=variant,
            group=ModelGroup.PRIORITY,  # Use appropriate enum
            task=ModelTask.CV_IMAGE_CLS,  # Use appropriate enum
            source=ModelSource.HUGGING_FACE,  # Use appropriate enum
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the model instance with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
        """
        model = MyModel(param1=self.param1, param2=self.param2)
        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the model with default settings.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
        """
        # Create inputs with default dtype
        inputs = torch.rand(1, 3, 224, 224)
        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)
        return inputs
```

```python
# mymodel/pytorch/__init__.py
from .loader import ModelLoader, ModelVariant
```

```python
# mymodel/__init__.py
# Import from the PyTorch implementation by default
from .pytorch import ModelLoader
```

## Supporting HuggingFace Models

For models available through the HuggingFace Transformers library, we create wrapper loader classes that delegate to the HF APIs. This approach leverages HuggingFace's infrastructure while providing a standardized interface.

Example of BERT implementation with variant support:

```python
# bert/pytorch/loader.py
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
)

class ModelLoader(ForgeModel):
    """BERT model loader implementation for question answering tasks."""

    # Dictionary of available model variants
    class ModelVariant(StrEnum):
        """Available model variants."""
        BASE = "base"
        LARGE = "large"

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="phiyodr/bert-base-finetuned-squad2",
            max_length=256,
        ),
        ModelVariant.LARGE: LLMModelConfig(
            pretrained_model_name="phiyodr/bert-large-finetuned-squad2",
            max_length=256,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LARGE

    # Sample inputs for inference
    context = 'Johann Joachim Winckelmann was a German art historian and archaeologist...'
    question = "What discipline did Winkelmann create?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant."""
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant."""
        return ModelInfo(
            model="bert",
            variant=variant,
            group=ModelGroup.PRIORITY,
            task=ModelTask.NLP_QA,  # Standardized task enum
            source=ModelSource.HUGGING_FACE,  # Standardized source enum
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant."""
        tokenizer_kwargs = {"padding_side": "left"}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override

        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        return self.tokenizer

    def load_model(self, dtype_override=None):
        """Load and return the BERT model instance for this instance's variant."""
        # Ensure tokenizer is loaded
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override

        model = AutoModelForQuestionAnswering.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        return model

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs for the BERT model with this instance's variant settings."""
        # Ensure tokenizer is initialized
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Get max_length from the variant config
        max_length = self._variant_config.max_length

        # Create tokenized inputs
        inputs = self.tokenizer.encode_plus(
            self.question,
            self.context,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        return inputs
```

## Consuming in Other Projects

When this repository is used as a git submodule or dependency in other projects, it can be imported using the project's specific import paths:

```python
# All models use the consistent ModelLoader class name
from third_party.tt_forge_models.yolov4.pytorch import ModelLoader
from third_party.tt_forge_models.bert.pytorch import ModelLoader

# To use multiple models in the same file, use aliases
from third_party.tt_forge_models.yolov4.pytorch import ModelLoader as YOLOv4Loader, ModelVariant as YOLOv4Variant
from third_party.tt_forge_models.bert.pytorch import ModelLoader as BertLoader, ModelVariant as BertVariant

# For base classes and utilities
from third_party.tt_forge_models.base import ForgeModel
