# tt-forge-models

A shared repository of model implementations that can be used across TT-Forge frontend repositories.

## Overview

This repository contains model implementations that are used for testing and benchmarking across various TT-Forge frontend repositories. The goal is to have a single source of truth for model implementations rather than duplicating code across different repositories.

## Structure

```
├── base.py                 # Base interface for all model loaders
├── yolov4/                 # YOLOv4 model implementation
│   ├── __init__.py         # Package initialization
│   ├── loader.py           # Public interface
│   └── src/                # Model implementation details
│       ├── yolov4.py       # Main model implementation
│       ├── downsample1.py  # Model components
│       └── ...
├── yolov3/
│   ├── __init__.py
│   ├── loader.py
│   └── src/
├── oft/
│   ├── __init__.py
│   ├── loader.py
│   └── src/
└── ...
```

## Usage

Each model in tt-forge-models follows the same interface pattern. Import the loader for your model and use it to load both the model and sample inputs:

```python
from tt_forge_models.yolov4 import YOLOv4Loader

# Load model
model = YOLOv4Loader.load_model(dtype=torch.bfloat16)

# Load sample inputs
inputs = YOLOv4Loader.load_inputs(dtype=torch.bfloat16)

# Use the model
outputs = model(inputs)
```

## Adding New Models

To add a new model:

1. Create a new directory for your model
2. Create a `loader.py` file that implements the ForgeModel interface
3. Create a `src/` directory for your model implementation files
4. Implement `load_model()` and `load_inputs()` methods in your loader

Example for a new model:

```python
# mymodel/loader.py
from tt_forge_models.base import ForgeModel
from .src.model import MyModel

class MyModelLoader(ForgeModel):
    @classmethod
    def load_model(cls, dtype=None, **kwargs):
        """Load and return the model instance."""
        model = MyModel(**kwargs)

        if dtype is not None:
            model = model.to(dtype)

        return model

    @classmethod
    def load_inputs(cls, dtype=None, **kwargs):
        """Load and return sample inputs for the model."""
        # Implementation here
        pass
```

## Supporting HuggingFace Models

For models that are available through the HuggingFace Transformers library, you can create loader classes that delegate to the HF APIs:

```python
# bert/loader.py
from tt_forge_models.base import ForgeModel

class BertLoader(ForgeModel):
    @classmethod
    def load_model(cls, model_name="bert-base-uncased", dtype=None, **kwargs):
        """Load a BERT model from Hugging Face."""
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_name, **kwargs)

        if dtype is not None:
            model = model.to(dtype)

        return model

    @classmethod
    def load_inputs(cls, batch_size=1, seq_length=128, dtype=None, **kwargs):
        """Generate sample inputs for BERT models."""
        import torch

        # Create input tensors
        input_ids = torch.randint(0, 30000, (batch_size, seq_length))
        attention_mask = torch.ones((batch_size, seq_length))

        if dtype is not None:
            input_ids = input_ids.to(dtype)
            attention_mask = attention_mask.to(dtype)

        return {"input_ids": input_ids, "attention_mask": attention_mask}
```

## Consuming in Other Projects

When this repository is used as a git submodule or dependency in other projects, it can be imported using the project's specific import paths:

```python
# For models
from tt_forge_models.yolov4 import YOLOv4Loader
from tt_forge_models.oft import OFTLoader

# For base classes and utilities
from tt_forge_models.base import ForgeModel
```
