# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
K-BERT model loader implementation for multi-task classification.
"""

import torch
from transformers import AutoModel, AutoTokenizer
from third_party.tt_forge_models.config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
    LLMModelConfig,
)
from third_party.tt_forge_models.base import ForgeModel


class ModelVariant(StrEnum):
    """Available K-BERT model variants for multi-task classification."""

    LQ_KBERT_BASE = "LQ_Kbert_Base"


class ModelLoader(ForgeModel):
    """K-BERT model loader implementation for multi-task classification."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.LQ_KBERT_BASE: LLMModelConfig(
            pretrained_model_name="LangQuant/LQ-Kbert-base",
            max_length=200,
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.LQ_KBERT_BASE

    def __init__(self, variant=None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional string specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

        pretrained_model_name = self._variant_config.pretrained_model_name
        self.model_name = pretrained_model_name
        self.sample_text = "비트코인 조정 후 반등, 투자심리 개선"
        self.max_length = self._variant_config.max_length
        self.tokenizer = None

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information for dashboard and metrics reporting.

        Args:
            variant_name: Optional variant name string. If None, uses 'base'.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant_name is None:
            variant_name = "base"
        return ModelInfo(
            model="K-BERT",
            variant=variant_name,
            group=ModelGroup.VULCAN,
            task=ModelTask.NLP_TEXT_CLS,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.TORCH,
        )

    def load_model(self, *, dtype_override=None, **kwargs):
        """Load K-BERT model for multi-task classification from Hugging Face.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            torch.nn.Module: The K-BERT model instance.
        """

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load pre-trained model from HuggingFace
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["torch_dtype"] = dtype_override
        model_kwargs |= kwargs

        model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True, **model_kwargs
        )
        self.model = model
        model.eval()
        return model

    def load_inputs(self, dtype_override=None):
        """Prepare sample input for K-BERT multi-task classification.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            dict: Input tensors that can be fed to the model.
        """
        if self.tokenizer is None:
            self.load_model(dtype_override=dtype_override)

        inputs = self.tokenizer(
            self.sample_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return inputs

    def decode_output(self, co_out):
        """Decode the model output for multi-task classification."""
        IDX2SENTI = {
            0: "strong_pos",
            1: "weak_pos",
            2: "neutral",
            3: "weak_neg",
            4: "strong_neg",
        }
        IDX2ACT = {
            0: "buy",
            1: "hold",
            2: "sell",
            3: "avoid",
            4: "info_only",
            5: "ask_info",
        }
        EMO_LIST = [
            "greed",
            "fear",
            "confidence",
            "doubt",
            "anger",
            "hope",
            "sarcasm",
        ]

        logits_senti = co_out["logits_senti"]
        logits_act = co_out["logits_act"]
        logits_emo = co_out["logits_emo"]
        pred_reg = co_out["pred_reg"]

        senti = int(logits_senti[0].argmax().item())
        act = int(logits_act[0].argmax().item())
        emo_p = torch.sigmoid(logits_emo[0]).tolist()
        reg = torch.clamp(pred_reg[0], 0, 1).tolist()
        emos = [EMO_LIST[j] for j, p in enumerate(emo_p) if p >= 0.5]

        print(f"Predicted Sentiment: {IDX2SENTI[senti]}")
        print(f"Predicted Action: {IDX2ACT[act]}")
        print(f"Predicted Emotions: {emos}")
        print(
            f"Certainty: {reg[0]:.3f}, Relevance: {reg[1]:.3f}, Toxicity: {reg[2]:.3f}"
        )
