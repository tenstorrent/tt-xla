# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SAP RPT-1-OSS model loader for tabular classification.

SAP RPT-1-OSS (formerly ConTextTab) is a tabular foundation model for
in-context learning on tabular data. It uses 2D attention (cross-column +
cross-row) with cell embeddings derived from sentence transformers.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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
    """Available SAP RPT model variants."""

    SAP_RPT_1_OSS = "SAP/sap-rpt-1-oss"


class SapRptWrapper(nn.Module):
    """Wrapper around RPT model that accepts tensor inputs for XLA tracing.

    The SAP RPT model normally uses a scikit-learn-style API (fit/predict)
    which is not compatible with XLA tracing. This wrapper accepts pre-computed
    cell embeddings and passes them through the model directly.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, cell_emb, attn_mask, output_mask):
        """Run forward pass with tensor inputs.

        Args:
            cell_emb: Cell embeddings tensor.
            attn_mask: Attention mask tensor.
            output_mask: Output mask tensor.

        Returns:
            Tensor: Model output predictions.
        """
        return self.model(
            cell_emb=cell_emb, attn_mask=attn_mask, output_mask=output_mask
        )


class ModelLoader(ForgeModel):
    """SAP RPT-1-OSS model loader for tabular classification."""

    _VARIANTS = {
        ModelVariant.SAP_RPT_1_OSS: ModelConfig(
            pretrained_model_name="SAP/sap-rpt-1-oss",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.SAP_RPT_1_OSS

    # Sample tabular data for binary classification (is_drama)
    _SAMPLE_TRAIN_DATA = {
        "title": [
            "The Shawshank Redemption",
            "The Dark Knight",
            "Forrest Gump",
            "The Matrix",
        ],
        "year": [1994, 2008, 1994, 1999],
        "genre": ["drama", "action", "drama", "sci-fi"],
    }
    _SAMPLE_TRAIN_LABELS = [1, 0, 1, 0]

    _SAMPLE_TEST_DATA = {
        "title": ["Inception", "The Godfather"],
        "year": [2010, 1972],
        "genre": ["sci-fi", "drama"],
    }

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._classifier = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="SAP-RPT-1-OSS",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load SAP RPT-1-OSS model wrapped for tensor-based inference.

        Returns:
            torch.nn.Module: The wrapped RPT model instance.
        """
        from sap_rpt_oss import SAP_RPT_OSS_Classifier

        self._classifier = SAP_RPT_OSS_Classifier(max_context_size=2048, bagging=1)

        X_train = pd.DataFrame(self._SAMPLE_TRAIN_DATA)
        y_train = np.array(self._SAMPLE_TRAIN_LABELS)
        self._classifier.fit(X_train, y_train)

        wrapper = SapRptWrapper(self._classifier.model)
        wrapper.eval()
        return wrapper

    def load_inputs(self, dtype_override=None):
        """Prepare sample inputs for the SAP RPT model.

        Returns:
            list: [cell_emb, attn_mask, output_mask] tensors.
        """
        if self._classifier is None:
            self.load_model(dtype_override=dtype_override)

        X_train = pd.DataFrame(self._SAMPLE_TRAIN_DATA)
        y_train = np.array(self._SAMPLE_TRAIN_LABELS)
        X_test = pd.DataFrame(self._SAMPLE_TEST_DATA)

        tokenized = self._classifier.tokenize(X_train, y_train, X_test)

        dtype = dtype_override if dtype_override is not None else torch.float32

        return [
            tokenized.cell_emb.to(dtype),
            tokenized.attn_mask,
            tokenized.output_mask,
        ]
