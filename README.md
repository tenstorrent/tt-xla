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

# Load model with default settings - no parameters needed
model = ModelLoader.load_model()

# Load sample inputs with default settings
inputs = ModelLoader.load_inputs()

# Use the model
outputs = model(inputs)
```

## Adding New Models

To add a new model:

1. Create a new top-level directory for your model (e.g., `mynewmodel/`)
2. Create a framework-specific subdirectory (e.g., `mynewmodel/pytorch/`)
3. Create a `loader.py` file within the framework directory that implements the ForgeModel interface with a class named `ModelLoader`
4. Create a `src/` directory for your model implementation files (if needed)
5. Create `__init__.py` files in both the top-level and framework directories
6. Implement parameter-free `load_model()` and `load_inputs()` methods

Example for a new model:

```python
# mymodel/pytorch/loader.py
from ...base import ForgeModel
from .src.model import MyModel

class ModelLoader(ForgeModel):
    # Shared configuration parameters as class variables
    param1 = "default_value"
    param2 = 42

    @classmethod
    def load_model(cls):
        """Load and return the model instance with default settings."""
        model = MyModel(param1=cls.param1, param2=cls.param2)
        return model.to(torch.bfloat16)

    @classmethod
    def load_inputs(cls):
        """Load and return sample inputs for the model with default settings."""
        # Create inputs with reasonable defaults
        inputs = torch.rand(1, 3, 224, 224)
        return inputs
```

```python
# mymodel/pytorch/__init__.py
from .loader import ModelLoader
```

```python
# mymodel/__init__.py
# Import from the PyTorch implementation by default
from .pytorch import ModelLoader
```

## Future Considerations

The new framework-aware directory structure is designed to support various future enhancements:

1. **Multiple Framework Implementations**: The same model could be implemented in PyTorch, TensorFlow, JAX, or other frameworks while maintaining a consistent interface.

2. **Model Variants**: Different versions of the same architecture (e.g., small, medium, large) could be implemented and selected via configuration.

## Supporting HuggingFace Models

For models available through the HuggingFace Transformers library, we create wrapper loader classes that delegate to the HF APIs. This approach leverages HuggingFace's infrastructure while providing a standardized interface.

Example of BERT implementation:

```python
# bert/pytorch/loader.py
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from ...base import ForgeModel

class ModelLoader(ForgeModel):
    # Shared configuration parameters
    model_name = "phiyodr/bert-large-finetuned-squad2"
    torch_dtype = torch.bfloat16
    context = 'Johann Joachim Winckelmann was a German art historian and archaeologist...'
    question = "What discipline did Winkelmann create?"

    @classmethod
    def load_model(cls):
        """Load a BERT model from Hugging Face."""
        model = AutoModelForQuestionAnswering.from_pretrained(
            cls.model_name, torch_dtype=cls.torch_dtype
        )
        return model

    @classmethod
    def load_inputs(cls):
        """Generate sample inputs for BERT models."""
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            cls.model_name, padding_side="left", torch_dtype=cls.torch_dtype
        )

        # Create tokenized inputs
        inputs = tokenizer.encode_plus(
            cls.question, cls.context,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=256,
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
from third_party.tt_forge_models.yolov4.pytorch import ModelLoader as YOLOv4Loader
from third_party.tt_forge_models.bert.pytorch import ModelLoader as BertLoader

# For base classes and utilities
from third_party.tt_forge_models.base import ForgeModel
