# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Mitra model loader implementation for tabular classification.

Mitra is a tabular foundation model for classification using an in-context
learning paradigm. It operates on support/query sets of tabular data using
a 12-layer Transformer with 2D attention (across observations and features).
"""
import torch
from typing import Optional

from ...base import ForgeModel
from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)


class ModelVariant(StrEnum):
    """Available Mitra model variants."""

    MITRA_CLASSIFIER = "autogluon/mitra-classifier"


class ModelLoader(ForgeModel):
    """Mitra model loader implementation for tabular classification."""

    _VARIANTS = {
        ModelVariant.MITRA_CLASSIFIER: ModelConfig(
            pretrained_model_name="autogluon/mitra-classifier",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.MITRA_CLASSIFIER

    # In-context learning parameters
    _N_FEATURES = 4
    _N_SUPPORT = 8
    _N_QUERY = 2
    _DIM_OUTPUT = 10

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="Mitra",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load Mitra model for tabular classification.

        Returns:
            torch.nn.Module: The Mitra Tab2D model instance.
        """
        from autogluon.tabular.models.mitra._internal.models.tab2d import Tab2D

        model_kwargs = {"device": "cpu"}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = Tab2D.from_pretrained(
            self._variant_config.pretrained_model_name, **model_kwargs
        )
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample in-context learning inputs for the Mitra model.

        Returns:
            list: [x_support, y_support, x_query, padding_features,
                   padding_obs_support, padding_obs_query] tensors.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32
        batch_size = 1

        # Support set: labeled examples for in-context learning
        x_support = torch.randn(
            batch_size, self._N_SUPPORT, self._N_FEATURES, dtype=dtype
        )
        y_support = torch.randint(
            0, self._DIM_OUTPUT, (batch_size, self._N_SUPPORT), dtype=torch.int64
        )

        # Query set: examples to classify
        x_query = torch.randn(batch_size, self._N_QUERY, self._N_FEATURES, dtype=dtype)

        # No padding - all features and observations are valid
        padding_features = torch.zeros(batch_size, self._N_FEATURES, dtype=torch.bool)
        padding_obs_support = torch.zeros(batch_size, self._N_SUPPORT, dtype=torch.bool)
        padding_obs_query = torch.zeros(batch_size, self._N_QUERY, dtype=torch.bool)

        return [
            x_support,
            y_support,
            x_query,
            padding_features,
            padding_obs_support,
            padding_obs_query,
        ]
