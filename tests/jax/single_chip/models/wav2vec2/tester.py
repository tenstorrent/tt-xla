# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
from datasets import load_dataset
from infra import ComparisonConfig, JaxModelTester, RunMode
from transformers import AutoProcessor, FlaxPreTrainedModel, FlaxWav2Vec2ForCTC


class Wav2Vec2Tester(JaxModelTester):
    """Tester for Wav2Vec2 model."""

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
        return FlaxWav2Vec2ForCTC.from_pretrained(self._model_path)

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        processor = AutoProcessor.from_pretrained(self._model_path)
        dataset = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )
        sample = dataset[0]["audio"]
        inputs = processor(
            sample["array"],
            sampling_rate=sample["sampling_rate"],
            return_tensors="jax",
        )
        return inputs
