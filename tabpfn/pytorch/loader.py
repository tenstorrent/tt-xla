# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
TabPFN v2 model loader implementation for tabular classification.

TabPFN (Prior-Data Fitted Network) is a transformer-based foundation model for
tabular data that uses in-context learning to perform classification. Rather than
fitting parameters, it processes training examples and test samples together
through attention mechanisms to make predictions.
"""
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
    """Available TabPFN model variants."""

    TABPFN_V2_CLF = "Prior-Labs/TabPFN-v2-clf"


class TabPFNWrapper(nn.Module):
    """Wrapper around TabPFN's internal transformer for XLA-compatible tracing.

    TabPFN uses in-context learning: training data and test data are concatenated
    and passed through a transformer in a single forward pass. This wrapper bakes
    in sample training data so the forward method accepts only test features.
    """

    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def forward(self, X_test):
        """Run TabPFN classification on test features.

        Args:
            X_test: Test features tensor (n_test_samples, n_features).

        Returns:
            Tensor: Class probability predictions (n_test_samples, n_classes).
        """
        probas = self.classifier.predict_proba(X_test.numpy())
        return torch.tensor(probas, dtype=X_test.dtype)


class ModelLoader(ForgeModel):
    """TabPFN v2 model loader for tabular classification."""

    _VARIANTS = {
        ModelVariant.TABPFN_V2_CLF: ModelConfig(
            pretrained_model_name="Prior-Labs/TabPFN-v2-clf",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.TABPFN_V2_CLF

    # Sample Iris-like dataset for multiclass classification
    _N_FEATURES = 4
    _N_TRAIN = 20
    _N_TEST = 5
    _N_CLASSES = 3

    def __init__(self, variant: Optional[ModelVariant] = None):
        super().__init__(variant)
        self._classifier = None

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        if variant is None:
            variant = cls.DEFAULT_VARIANT
        return ModelInfo(
            model="TabPFN-v2",
            variant=variant,
            group=ModelGroup.VULCAN,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load TabPFN v2 classifier and fit it with sample training data.

        Returns:
            torch.nn.Module: The wrapped TabPFN model instance.
        """
        from tabpfn import TabPFNClassifier

        self._classifier = TabPFNClassifier()

        # Generate deterministic sample training data
        torch.manual_seed(42)
        X_train = torch.randn(self._N_TRAIN, self._N_FEATURES)
        y_train = torch.arange(self._N_CLASSES).repeat(
            self._N_TRAIN // self._N_CLASSES + 1
        )[: self._N_TRAIN]

        self._classifier.fit(X_train.numpy(), y_train.numpy())

        wrapper = TabPFNWrapper(self._classifier)
        wrapper.eval()
        return wrapper

    def load_inputs(self, dtype_override=None):
        """Prepare sample test inputs for the TabPFN model.

        Returns:
            list: [X_test] tensor of test features.
        """
        dtype = dtype_override if dtype_override is not None else torch.float32

        torch.manual_seed(123)
        X_test = torch.randn(self._N_TEST, self._N_FEATURES, dtype=dtype)

        return [X_test]
