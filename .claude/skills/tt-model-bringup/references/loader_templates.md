# Model Loader Templates

Copy-paste templates for common model bringup scenarios. Replace all `<placeholders>`.

## Template 1: HuggingFace Text Model (Causal LM)

Use for models loaded via `AutoModelForCausalLM` (GPT-2, Llama, Qwen, Mistral, Falcon, etc.).

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
<ModelName> model loader implementation for causal language modeling.
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Optional

from ...base import ForgeModel
from ...config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available <ModelName> model variants for causal LM."""
    <VARIANT_ENUM_1> = "<variant_value_1>"
    <VARIANT_ENUM_2> = "<variant_value_2>"


class ModelLoader(ForgeModel):
    """<ModelName> model loader implementation for causal language modeling tasks."""

    _VARIANTS = {
        ModelVariant.<VARIANT_ENUM_1>: LLMModelConfig(
            pretrained_model_name="<org>/<model-name-1>",
            max_length=128,
        ),
        ModelVariant.<VARIANT_ENUM_2>: LLMModelConfig(
            pretrained_model_name="<org>/<model-name-2>",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.<VARIANT_ENUM_1>

    sample_text = "<A representative prompt for this model>"

    def __init__(self, variant: Optional[ModelVariant] = None, num_layers: Optional[int] = None):
        super().__init__(variant)
        self.tokenizer = None
        self.config = None
        self.model = None
        self.num_layers = num_layers

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="<ModelName>",
            variant=variant,
            group=ModelGroup.<GROUP>,         # RED, PRIORITY, or GENERALITY
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def _load_tokenizer(self, dtype_override=None):
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["torch_dtype"] = dtype_override
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name, **tokenizer_kwargs
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        pretrained_model_name = self._variant_config.pretrained_model_name
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        if self.num_layers is not None:
            config = AutoConfig.from_pretrained(pretrained_model_name)
            config.num_hidden_layers = self.num_layers
            model_kwargs["config"] = config

        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name, **model_kwargs)
        model.eval()
        self.model = model
        self.config = model.config
        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        if self.tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        max_length = self._variant_config.max_length
        inputs = self.tokenizer(
            self.sample_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        for key in inputs:
            if torch.is_tensor(inputs[key]):
                inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

        return inputs

    def load_config(self):
        self.config = AutoConfig.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        return self.config
```

## Template 2: HuggingFace Text Model (Question Answering / Classification)

Use for task-specific models like BERT QA, DistilBERT classification, etc.

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
<ModelName> model loader implementation for <task_description>.
"""
import torch
from transformers import Auto<TaskModel>, AutoTokenizer
from typing import Optional

from ....base import ForgeModel        # Note: four dots for task-scoped layout
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
    """Available <ModelName> model variants."""
    BASE = "Base"
    LARGE = "Large"


class ModelLoader(ForgeModel):
    """<ModelName> model loader implementation for <task> tasks."""

    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="<org>/<model-base>",
            max_length=384,
        ),
        ModelVariant.LARGE: LLMModelConfig(
            pretrained_model_name="<org>/<model-large>",
            max_length=384,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    # Sample data
    context = "<representative context>"
    question = "<representative question>"

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="<ModelName>",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.<TASK>,             # NLP_QA, NLP_TEXT_CLS, etc.
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._variant_config.pretrained_model_name
        )
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Auto<TaskModel>.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.question,
            self.context,
            max_length=self._variant_config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs

    def decode_output(self, co_out):
        """Decode the model output."""
        inputs = self.load_inputs()
        # <model-specific output decoding>
        pass
```

## Template 3: Custom PyTorch Vision Model

Use for models with custom source code (not from HuggingFace), typically CV models.

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
<ModelName> model loader implementation.
"""
import torch
from torchvision import transforms
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from .src.<model_module> import <ModelClass>


class ModelVariant(StrEnum):
    """Available <ModelName> model variants."""
    BASE = "Base"


class ModelLoader(ForgeModel):
    """<ModelName> model loader implementation."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="",  # Not used for custom models
        )
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self.model = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="<ModelName>",
            variant=variant,
            group=ModelGroup.<GROUP>,
            task=ModelTask.<TASK>,             # CV_IMAGE_CLS, CV_OBJECT_DET, etc.
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        model = <ModelClass>(<constructor_args>)

        # Load pretrained weights if available.
        # When using get_file(), ensure IRD_LF_CACHE is set:
        #   export IRD_LF_CACHE=http://aus2-lfcache.aus2.tenstorrent.com/
        # weights_path = get_file("<path_to_weights>")
        # state_dict = torch.load(weights_path, map_location="cpu")
        # model.load_state_dict(state_dict)

        model.eval()
        self.model = model

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        # For vision models: create a preprocessed image tensor
        inputs = torch.rand(batch_size, 3, 224, 224)  # Adjust dimensions

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
```

## Template 4: TorchVision / TIMM Model

Use for models from `torchvision.models` or `timm`.

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
<ModelName> model loader implementation.
"""
import torch
import torchvision.models as models  # or: import timm
from torchvision import transforms
from typing import Optional

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available <ModelName> model variants."""
    DEFAULT = "Default"


class ModelLoader(ForgeModel):
    """<ModelName> model loader implementation."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(
            pretrained_model_name="<model_function_name>",
        )
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None):
        return ModelInfo(
            model="<ModelName>",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.TORCHVISION,    # or ModelSource.TIMM
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        # TorchVision:
        model = models.<model_fn>(pretrained=True, **kwargs)
        # Or TIMM:
        # model = timm.create_model("<model_name>", pretrained=True, **kwargs)

        model.eval()

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        inputs = torch.rand(batch_size, 3, 224, 224)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
```

## Template 5: EasyDeL JAX Model (Causal LM)

Use for JAX-based LLMs loaded via EasyDeL's `AutoEasyDeLModelForCausalLM`. This is the
standard pattern for JAX causal LM loaders (Llama, Qwen, Phi, Falcon, Mistral, Mamba, GPT-2, GPT-J).

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
<ModelName> model loader implementation for causal language modeling using EasyDeL/JAX.
"""
from typing import Optional

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
    Parallelism,
)
from ....base import ForgeModel

import flax.nnx as nnx
from jax.sharding import PartitionSpec
import jax.numpy as jnp
import numpy as np


class ModelVariant(StrEnum):
    """Available <ModelName> model variants for causal language modeling."""
    <VARIANT_ENUM_1> = "<variant_value_1>"
    <VARIANT_ENUM_2> = "<variant_value_2>"


class ModelLoader(ForgeModel):
    """<ModelName> model loader implementation for causal LM tasks using EasyDeL."""

    _VARIANTS = {
        ModelVariant.<VARIANT_ENUM_1>: LLMModelConfig(
            pretrained_model_name="<org>/<model-name-1>",
            max_length=128,
        ),
        ModelVariant.<VARIANT_ENUM_2>: LLMModelConfig(
            pretrained_model_name="<org>/<model-name-2>",
            max_length=128,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.<VARIANT_ENUM_1>

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="<ModelName>",
            variant=variant,
            group=ModelGroup.<GROUP>,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.EASYDEL,
            framework=Framework.JAX,
        )

    def __init__(self, variant=None):
        super().__init__(variant)
        self.input_text = "<A representative prompt for this model>"
        self.max_length = self._variant_config.max_length
        self.tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    def _load_tokenizer(self, dtype_override=None):
        from transformers import AutoTokenizer
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )
        return self.tokenizer

    def load_model(self, *, dtype_override=None, **kwargs):
        from easydel import AutoEasyDeLModelForCausalLM

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override
        model_kwargs |= kwargs

        partition_rules = ((r".*", PartitionSpec()),)
        model = AutoEasyDeLModelForCausalLM.from_pretrained(
            self._model_name, partition_rules=partition_rules, **model_kwargs
        )
        return model

    def load_inputs(self, dtype_override=None, mesh=None):
        if mesh is not None:
            num_devices = np.prod(list(mesh.shape.values())) if mesh.shape else 1
            batch_size = 8
            if batch_size % num_devices != 0:
                batch_size = num_devices * (batch_size // num_devices + 1)
        else:
            batch_size = 8

        if self.tokenizer is None:
            self._load_tokenizer(dtype_override)

        inputs = self.tokenizer(self.input_text, return_tensors="jax")
        input_ids = jnp.repeat(inputs.input_ids, batch_size, axis=0)
        return {"input_ids": input_ids}

    def get_input_activations_partition_spec(self, mesh, parallelism, axis_name="X"):
        if (
            parallelism.name == Parallelism.TENSOR_PARALLEL.name
            or np.prod(list(mesh.shape.values())) == 1
        ):
            return (PartitionSpec(),)
        return (PartitionSpec(axis_name),)

    def load_parameters_partition_spec(
        self,
        model_for_multichip,
        parallelism,
        axis_name="X",
        cpu_mesh=None,
        input_activations_partition_specs=None,
        inputs=None,
        dtype_override=None,
    ):
        state = nnx.split(model_for_multichip)[1]

        if (
            parallelism.name == Parallelism.DATA_PARALLEL.name
            or parallelism.name == Parallelism.SINGLE_DEVICE.name
        ):
            partition_rules = ((r".*", PartitionSpec()),)
        else:
            from easydel.modules.<arch> import <Arch>Config
            arch_config = <Arch>Config()
            partition_rules = arch_config.get_partition_rules()

        from infra.utilities import make_easydel_parameters_partition_specs
        return make_easydel_parameters_partition_specs(
            model_state=state, partition_rules=partition_rules, axis_name=axis_name
        )
```

## Template 6: Flax/Linen JAX Model (Non-EasyDeL)

Use for JAX models that use Flax Linen directly (custom architectures like MNIST, AlexNet)
without EasyDeL. Multi-chip support uses `make_flax_linen_parameters_partition_specs_on_cpu`.

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
<ModelName> model loader implementation for <task> using Flax/JAX.
"""
from typing import Optional

from ....config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
    Parallelism,
)
from ....base import ForgeModel

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec
from flax import linen as nn


class ModelVariant(StrEnum):
    """Available model variants."""
    DEFAULT = "Default"
    MULTICHIP = "Multichip"


class ModelLoader(ForgeModel):
    """Model loader using Flax Linen."""

    _VARIANTS = {
        ModelVariant.DEFAULT: ModelConfig(pretrained_model_name=""),
        ModelVariant.MULTICHIP: ModelConfig(pretrained_model_name=""),
    }

    DEFAULT_VARIANT = ModelVariant.DEFAULT

    @classmethod
    def _get_model_info(cls, variant=None):
        return ModelInfo(
            model="<ModelName>",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.<TASK>,
            source=ModelSource.CUSTOM,
            framework=Framework.JAX,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        # Return a Flax Linen module
        model = <YourFlaxModule>()
        return model

    def load_inputs(self, dtype_override=None, mesh=None):
        inputs = jnp.ones((batch_size, *input_shape))
        return (inputs,)

    def get_input_activations_partition_spec(self, mesh, parallelism, axis_name="X"):
        if (
            parallelism.name == Parallelism.TENSOR_PARALLEL.name
            or np.prod(list(mesh.shape.values())) == 1
        ):
            return (PartitionSpec(),)
        return (PartitionSpec(axis_name),)

    def load_parameters_partition_spec(
        self, model_for_multichip, parallelism, axis_name="X",
        cpu_mesh=None, input_activations_partition_specs=None,
        inputs=None, dtype_override=None,
    ):
        from infra.utilities import (
            make_flax_linen_parameters_partition_specs_on_cpu,
            initialize_flax_linen_parameters_on_cpu,
        )
        init_params = initialize_flax_linen_parameters_on_cpu(
            model=model_for_multichip, cpu_mesh=cpu_mesh, inputs=inputs
        )
        return make_flax_linen_parameters_partition_specs_on_cpu(
            params=init_params, partition_rules=((r".*", PartitionSpec()),)
        )
```

## __init__.py Templates

### Top-level `__init__.py`

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from .pytorch import ModelLoader   # or: from .jax import ModelLoader
```

### Framework-level `__init__.py`

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from .loader import ModelLoader, ModelVariant
```
