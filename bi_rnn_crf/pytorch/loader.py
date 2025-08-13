# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BiRNN-CRF model loader implementation for sequence tagging
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
    """Available BiRNN-CRF model variants."""

    LSTM = "lstm"
    GRU = "gru"


class ModelLoader(ForgeModel):
    """BiRNN-CRF model loader implementation for sequence tagging tasks."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.LSTM: ModelConfig(
            pretrained_model_name="bi-rnn-crf-lstm",
        ),
        ModelVariant.GRU: ModelConfig(
            pretrained_model_name="bi-rnn-crf-gru",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LSTM

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which RNN type to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        # Model configuration
        self.model_config = {
            "rnn_type": self._variant.value.lower(),
            "vocab_size": 30000,
            "tagset_size": 20,
            "embedding_dim": 256,
            "hidden_dim": 512,
            "num_rnn_layers": 2,
        }

        self.rnn_type = self.model_config["rnn_type"]

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which RNN type to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        return ModelInfo(
            model=f"BiRnnCrf-{variant.value.upper()}",
            variant=variant,
            group=ModelGroup.RED,
            task=ModelTask.NLP_TOKEN_CLS,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self):
        """Load and return the BiRNN-CRF model instance for this instance's variant.

        Returns:
            torch.nn.Module: The BiRNN-CRF model instance for sequence tagging.
        """
        from bi_lstm_crf import BiRnnCrf

        # Create the model with random weights
        model = BiRnnCrf(
            vocab_size=self.model_config["vocab_size"],
            tagset_size=self.model_config["tagset_size"],
            embedding_dim=self.model_config["embedding_dim"],
            hidden_dim=self.model_config["hidden_dim"],
            num_rnn_layers=self.model_config["num_rnn_layers"],
            rnn=self.model_config["rnn_type"],
        )

        return model

    def load_inputs(self, batch_size=4):
        """Load and return sample inputs for the BiRNN-CRF model with this instance's variant settings.

        Args:
            batch_size: Optional batch size to override the default batch size of 4.

        Returns:
            torch.Tensor: Input tensor (token IDs) that can be fed to the model.
        """
        # Generate random token ids within vocabulary range
        seq_length = 16
        input_ids = torch.randint(
            0, self.model_config["vocab_size"], (batch_size, seq_length)
        )

        return input_ids

    # TODO - Verify this function correct (was AI_GENERATED)
    def decode_output(self, outputs):
        """Helper method to decode model outputs into human-readable format.

        Args:
            outputs: Model output from a forward pass (emissions, best_tag_sequence)

        Returns:
            str: Formatted output information
        """
        emissions, best_tag_sequence = outputs

        return f"""
        BiRNN-CRF Output:
          - Emissions shape: {emissions.shape}
          - Best tag sequence length: {len(best_tag_sequence[0])}
          - RNN type: {self.model_config["rnn_type"].upper()}
        """
