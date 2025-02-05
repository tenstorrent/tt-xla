# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
from infra import ComparisonConfig, ModelTester, RunMode
from transformers import FlaxResNetForImageClassification


class ResNetTester(ModelTester):
    "Tester for ResNet family of models."

    def __init__(
        self,
        model_name: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_name = model_name
        super().__init__(comparison_config, run_mode)

    # @override
    def _get_model(self):
        return FlaxResNetForImageClassification.from_pretrained(
            self._model_name, from_pt=True
        )  # only resnet-50 has a flax checkpoint

    # @override
    def _get_input_activations(self):
        data = jax.random.uniform(jax.random.PRNGKey(0), (1, 3, 224, 224))
        return data

    # @override
    def _get_forward_method_kwargs(self):
        assert hasattr(self._model, "params")
        return {
            "params": self._model.params,
            "pixel_values": self._get_input_activations(),
        }

    # @override
    def _get_static_argnames(self):
        return ["train"]
