# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Whisper model loader implementation for audio classification.
"""

from typing import Optional

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
import flax.nnx as nnx
from jax.sharding import PartitionSpec
import jax.numpy as jnp
import numpy as np


class ModelVariant(StrEnum):
    """Available Whisper model variants."""

    BASE = "base"
    MEDIUM = "medium"
    LARGE_V3 = "large_v3"


class ModelLoader(ForgeModel):
    """Whisper model loader implementation for audio classification."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(
            pretrained_model_name="openai/whisper-base",
        ),
        ModelVariant.MEDIUM: ModelConfig(
            pretrained_model_name="openai/whisper-medium",
        ),
        ModelVariant.LARGE_V3: ModelConfig(
            pretrained_model_name="openai/whisper-large-v3",
        ),
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """

        super().__init__(variant)
        self._processor = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Method for getting model info with validated variant.
        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        Returns:
            ModelInfo: Information about the model and variant
        """

        # Use the provided variant or fall back to default
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="whisper",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.AUDIO_CLS,
            source=ModelSource.EASYDEL,
            framework=Framework.JAX,
        )

    def _load_processor(self, dtype_override=None):
        """Load audio processor for the current variant.

        Args:
            dtype_override: Optional dtype to override the processor's default dtype.

        Returns:
            processor: The loaded audio processor instance
        """

        from transformers import WhisperProcessor

        # Initialize processor with dtype_override if provided
        processor_kwargs = {}
        if dtype_override is not None:
            processor_kwargs["dtype"] = dtype_override

        # Load the processor
        self._processor = WhisperProcessor.from_pretrained(
            self._model_name, **processor_kwargs
        )

        return self._processor

    def load_model(self, dtype_override=None):
        """Load and return the Whisper model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The loaded model instance
        """

        from easydel import AutoEasyDeLModelForSpeechSeq2Seq

        # Initialize model kwargs
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        partition_rules = ((r".*", PartitionSpec()),)

        # Load the model
        model = AutoEasyDeLModelForSpeechSeq2Seq.from_pretrained(
            self._model_name, partition_rules=partition_rules, **model_kwargs
        )

        return model

    def load_inputs(self, dtype_override=None, mesh=None):
        """Load and return sample inputs for the Whisper model with this instance's variant settings.
        Args:
            dtype_override: Optional dtype to override the model's default dtype.
            mesh: Optional device mesh for sharding (DataParallel mode).
        Returns:
            inputs: Input tensors that can be fed to the model.
        """

        if mesh is not None:
            # For multi-device, use a fixed batch size that's divisible by device count
            # This matches the original test which used batch_size=8
            num_devices = np.prod(list(mesh.shape.values())) if mesh.shape else 1
            batch_size = 8  # Fixed batch size, will be sharded across devices
            # Ensure batch size is divisible by number of devices
            if batch_size % num_devices != 0:
                batch_size = num_devices * (batch_size // num_devices + 1)
        else:
            # Default to 8 for single device too, for consistency
            batch_size = 8

        from datasets import load_dataset
        from transformers import WhisperConfig

        # Ensure processor is initialized
        if self._processor is None:
            self._load_processor(dtype_override=dtype_override)

        dataset = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )
        whisper_config = WhisperConfig.from_pretrained(self._model_name)
        sample = dataset[0]["audio"]
        inputs = self._processor(
            sample["array"],
            sampling_rate=sample["sampling_rate"],
            return_tensors="jax",
        )
        inputs["decoder_input_ids"] = jnp.array(
            [[whisper_config.decoder_start_token_id]]
        )
        return inputs

    def get_input_activations_partition_spec(self, mesh, axis_name="X"):
        """Get partition specification for input activations.

        Args:
            mesh: The device mesh for sharding.
            axis_name: Name of the sharding axis.

        Returns:
            PartitionSpec for input activations (sharded on batch dimension)
        """
        if np.prod(list(mesh.shape.values())) == 1:
            return (PartitionSpec(), PartitionSpec())

        return (PartitionSpec(axis_name), PartitionSpec(axis_name))

    def load_parameters_partition_spec(
        self,
        model_for_multichip=None,
        cpu_mesh=None,
        input_activations_partition_specs=None,
        inputs=None,
        dtype_override=None,
    ):
        # Get the model state
        state = nnx.split(model_for_multichip)[1]

        partition_rules = ((r".*", PartitionSpec()),)

        from infra.utilities import make_easydel_parameters_partition_specs

        return make_easydel_parameters_partition_specs(
            model_state=state, partition_rules=partition_rules
        )
