# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import torch
from huggingface_hub import hf_hub_download
from infra import ComparisonConfig, ModelTester, RunMode
from safetensors import safe_open
from transformers import (
    FlaxResNetForImageClassification,
    ResNetConfig,
    FlaxPreTrainedModel,
)
from typing import List, Tuple, Dict, Union, Sequence

from tests.jax.models.model_utils import torch_statedict_to_pytree


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

    @staticmethod
    def _get_renaming_patterns(name) -> List[Tuple[str, str]]:
        PATTERNS = [
            (r"convolution.weight", r"convolution.kernel"),
            (r"normalization.running_mean", r"normalization.mean"),
            (r"normalization.running_var", r"normalization.var"),
            (r"normalization.weight", r"normalization.scale"),
            (r"classifier\.(\d+).weight", r"classifier.\1.kernel"),
        ]

        if name in ("resnet-18", "resnet-34"):
            PATTERNS.append((r"layer\.(\d+)\.", r"layer.layer_\1."))

        return PATTERNS

    @staticmethod
    def _get_banned_subkeys() -> List[str]:
        return ["num_batches_tracked"]

    @staticmethod
    def _download_weights(
        model_name: str,
    ) -> Union[Dict[str, jax.Array], Dict[str, torch.Tensor]]:
        filename = "model.safetensors"
        if "resnet-101" in model_name:
            filename = "pytorch_model.bin"

        ckpt_path = hf_hub_download(repo_id=model_name, filename=filename)
        if filename == "model.safetensors":
            with safe_open(ckpt_path, framework="flax", device="cpu") as f:
                return {key: f.get_tensor(key) for key in f.keys()}
        else:  # filename == "pytorch_model.bin"
            return torch.load(ckpt_path, map_location="cpu")

    # @override
    def _get_model(self) -> FlaxPreTrainedModel:
        model_variant = self._model_name.split("/")[-1]
        # Resnet-50 has a flax checkpoint on HF, so we can just load it directly.
        if model_variant == "resnet-50":
            return FlaxResNetForImageClassification.from_pretrained(self._model_name)

        # We would ideally rely on 'from_pt' functionality in HF,
        # however, it is broken for resnet.
        # All the weights fail to load because of a naming mismatch.
        # We have to load the weights manually and apply a few conversions.
        model_config = ResNetConfig.from_pretrained(self._model_name)
        model = FlaxResNetForImageClassification(model_config)

        state_dict = self._download_weights(self._model_name)

        variables = torch_statedict_to_pytree(
            state_dict,
            patterns=self._get_rename_patterns(model_variant),
            banned_subkeys=self._get_banned_subkeys(),
        )

        model.params = variables

        return model

    # @override
    def _get_input_activations(self) -> jax.Array:
        return jax.random.uniform(jax.random.PRNGKey(0), (1, 3, 224, 224))

    # @override
    def _get_forward_method_kwargs(self) -> Dict[str, jax.Array]:
        assert hasattr(self._model, "params")
        return {
            "params": self._model.params,
            "pixel_values": self._get_input_activations(),
        }

    # @override
    def _get_static_argnames(self) -> Sequence[str]:
        return ["train"]
