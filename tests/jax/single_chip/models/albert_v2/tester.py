# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
from infra import ComparisonConfig, JaxModelTester, RunMode
from transformers import AutoTokenizer, FlaxAlbertForMaskedLM, FlaxPreTrainedModel


class AlbertV2Tester(JaxModelTester):
    """Tester for Albert model on a masked language modeling task."""

    def __init__(
        self,
        model_path: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_path = model_path
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self) -> FlaxPreTrainedModel:
        return FlaxAlbertForMaskedLM.from_pretrained(self._model_path)

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        inputs = tokenizer("Hello [MASK].", return_tensors="jax")
        return inputs

    # @ override
    def _get_static_argnames(self):
        return ["train"]


# TODO(stefan): Add testers for Albert when used as a question answering or sentiment analysis model.
