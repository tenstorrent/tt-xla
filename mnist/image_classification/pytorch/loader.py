# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
MNIST model loader implementation for image classification.
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    ModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MNISTCNNDropoutModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.dropout2(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class MNISTCNNNoDropoutModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding="same")
        self.bn1 = nn.BatchNorm2d(32, eps=1e-5)

        self.conv2 = nn.Conv2d(32, 64, 3, padding="same")
        self.bn2 = nn.BatchNorm2d(64, eps=1e-5)

        self.fc1 = nn.Linear(7 * 7 * 64, 256)
        self.bn3 = nn.BatchNorm1d(256, eps=1e-5)

        self.fc2 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128, eps=1e-5)

        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = torch.flatten(x, 1)  # Flatten all dimensions except batch

        x = self.fc1(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = F.softmax(x, dim=1)

        return x


class ModelVariant(StrEnum):
    """Available MNIST model variants for image classification."""

    CNN_DROPOUT = "cnn_dropout"
    CNN_NODROPOUT = "cnn_nodropout"


class ModelLoader(ForgeModel):
    """MNIST model loader implementation for image classification."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.CNN_DROPOUT: ModelConfig(
            pretrained_model_name="mnist_cnn_dropout"
        ),
        ModelVariant.CNN_NODROPOUT: ModelConfig(
            pretrained_model_name="mnist_cnn_nodropout"
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.CNN_DROPOUT

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'cnn_dropout'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "cnn_dropout"
        return ModelInfo(
            model="MNIST",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load MNIST model for image classification.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use bfloat16.

        Returns:
            torch.nn.Module: The MNIST model instance.
        """
        if self._variant == ModelVariant.CNN_DROPOUT:
            model = MNISTCNNDropoutModel()
        elif self._variant == ModelVariant.CNN_NODROPOUT:
            model = MNISTCNNNoDropoutModel()
        else:
            raise ValueError(f"Unknown variant: {self._variant}")

        # Apply dtype override or default to bfloat16
        if dtype_override is not None:
            model = model.to(dtype=dtype_override)
        else:
            model = model.to(dtype=torch.bfloat16)

        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for MNIST image classification.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use bfloat16.

        Returns:
            torch.Tensor: Input tensor that can be fed to the model.
        """
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.MNIST(
            root="./data", train=False, transform=transform, download=True
        )
        dataloader = DataLoader(test_dataset, batch_size=2)
        test_input, _ = next(iter(dataloader))

        if dtype_override is not None:
            test_input = test_input.to(dtype=dtype_override)
        else:
            test_input = test_input.to(dtype=torch.bfloat16)

        return test_input
