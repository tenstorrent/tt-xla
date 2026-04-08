# Model Loader Templates

Copy-paste templates for common model bringup scenarios. Replace all `<placeholders>`.

## Template 1: HuggingFace Text Model (Causal LM)

Use for models loaded via `AutoModelForCausalLM` (GPT-2, Llama, Qwen, Mistral, Falcon, etc.).

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
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
    <VARIANT_ENUM_1> = "<variant_value_1>"
    <VARIANT_ENUM_2> = "<variant_value_2>"


class ModelLoader(ForgeModel):

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
    BASE = "Base"
    LARGE = "Large"


class ModelLoader(ForgeModel):

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
    BASE = "Base"


class ModelLoader(ForgeModel):

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
    DEFAULT = "Default"


class ModelLoader(ForgeModel):

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
        inputs = torch.rand(batch_size, 3, 224, 224)

        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs
```

## __init__.py Templates

### Top-level `__init__.py`

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from .pytorch import ModelLoader
```

### Framework-level `__init__.py`

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from .loader import ModelLoader, ModelVariant
```
