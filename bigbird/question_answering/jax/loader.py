# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""BigBird model loader implementation for question answering."""

from typing import Optional

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.jax_utils import cast_hf_model_to_type


class ModelVariant(StrEnum):
    """Available BigBird model variants for question answering."""

    BASE = "base"
    LARGE = "large"


class ModelLoader(ForgeModel):
    """BigBird model loader implementation for question answering."""

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.BASE: LLMModelConfig(
            pretrained_model_name="google/bigbird-base-trivia-itc",
        ),
        ModelVariant.LARGE: LLMModelConfig(
            pretrained_model_name="google/bigbird-roberta-large",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.BASE

    context = """Walter Bruce Willis (born March 19, 1955) is an American actor, producer, and singer. His career began on the Off-Broadway stage and then in television in the 1980s,  most notably as
        David Addison in Moonlighting (1985–1989). He is known for his role of John McClane in the Die Hard series. He has appeared in over 60 films, including Color of Night (1994), Pulp Fiction (1994),
        12 Monkeys (1995), The Fifth Element (1997), Armageddon (1998), The Sixth Sense (1999), Unbreakable (2000), Sin City (2005), Red (2010), The Expendables 2 (2012), and Looper (2012).Willis married actress
        Demi Moore in 1987, and they had three daughters, including Rumer, before their divorce in 2000. Since 2009, he has been married to model Emma Heming, with whom he has two daughters.Willis was born Walter
        Bruce Willis on March 19, 1955 in the town of Idar-Oberstein, West Germany. His father, David Willis (1929-2009), was an American soldier. His mother, Marlene,  was German, born in Kassel."""

    question = "What is Bruce Willis' real first name?"

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information.

        Args:
            variant_name: Optional variant name string.  If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="bigbird",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_QA,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional dtype to override the tokenizer's default dtype.
                            If not provided, the tokenizer will use its default dtype (typically float32).

        Returns:
            The loaded tokenizer instance.
        """

        # Initialize tokenizer with dtype override if specified
        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        from transformers import AutoTokenizer

        # Load the tokenizer (AutoTokenizer for QA variants)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )

        return self._tokenizer

    def load_model(self, dtype_override=None):
        """Load and return BigBird model for question answering from Hugging Face.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            The loaded model instance.
        """

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        # Load the model with dtype override if specified
        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        from transformers import FlaxBigBirdForQuestionAnswering

        # For input_sequence_length < 1024, original_full attention type is used.
        # Ref : https://huggingface.co/docs/transformers/en/model_doc/big_bird#notes
        model = FlaxBigBirdForQuestionAnswering.from_pretrained(
            self._model_name, attention_type="original_full", **model_kwargs
        )

        # Cast the model to the dtype_override if provided
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load inputs for the BigBird model.

        Args:
            dtype_override: Optional dtype to override the inputs' default dtype.
                            If not provided, the inputs will use its default dtype (typically float32).

        Returns:
            The loaded inputs.
        """

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self._tokenizer(
            self.question,
            self.context,
            return_tensors="jax",
        )

        return inputs
