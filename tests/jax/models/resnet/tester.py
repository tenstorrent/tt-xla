# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import re

import flax.traverse_util
import jax
import jax.numpy as jnp
import torch
from huggingface_hub import hf_hub_download
from infra import ComparisonConfig, ModelTester, RunMode
from safetensors import safe_open
from transformers import FlaxResNetForImageClassification, ResNetConfig


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
        model_variant = self._model_name.split("/")[-1]
        # resnet-50 has a flax checkpoint on HF, so we can just load it directly
        if model_variant == "resnet-50":
            return FlaxResNetForImageClassification.from_pretrained(self._model_name)

        # for other variants, we would ideally rely on 'from_pt' functionality in HF,
        # however, it is broken, there is a naming mismatch between the checkpoints and the model so all the weights fail to load
        # so we instantiate the model from config, load the checkpoint manually, fix up the keys and load weights into the model
        model_config = ResNetConfig.from_pretrained(self._model_name)
        model = FlaxResNetForImageClassification(model_config)

        # download the checkpoint, we prefer safetensors if available
        # resnet-101 is the only one that doesn't have a .safetensors either,
        # so for it we have to use the pytorch checkpoint
        filename = "model.safetensors"
        if model_variant == "resnet-101":
            filename = "pytorch_model.bin"
        # load the checkpoint
        ckpt_path = hf_hub_download(repo_id=self._model_name, filename=filename)
        variables = {}
        if filename == "model.safetensors":
            with safe_open(ckpt_path, framework="flax", device="cpu") as f:
                for key in f.keys():
                    variables[key] = f.get_tensor(key)
        else:  # filename == "pytorch_model.bin"
            variables = torch.load(ckpt_path, map_location="cpu")
            for k, v in variables.items():
                variables[k] = jnp.array(v)

        # fix up the keys
        PATTERNS = [
            (r"convolution.weight", r"convolution.kernel"),
            (r"normalization.running_mean", r"normalization.mean"),
            (r"normalization.running_var", r"normalization.var"),
            (r"normalization.weight", r"normalization.scale"),
            (r"classifier\.(\d+).weight", r"classifier.\1.kernel"),
        ]

        if model_variant in ("resnet-18", "resnet-34"):
            # for whatever reason, 18 and 34 have a different naming scheme
            PATTERNS.append((r"layer\.(\d+)\.", r"layer.layer_\1."))

        def is_banned_key(key: str) -> bool:
            return "num_batches_tracked" in key

        def rewrite_key(key: str) -> str:
            is_batch_stat = "running_" in key
            prefix = "batch_stats." if is_batch_stat else "params."
            for pattern in PATTERNS:
                key = re.sub(pattern[0], pattern[1], key)
            return prefix + key

        def process_value(k: str, v) -> jnp.ndarray:
            if "kernel" in k:
                if len(v.shape) == 2:
                    return jnp.transpose(v)
                if len(v.shape) == 4:
                    return jnp.transpose(v, (2, 3, 1, 0))
            return v

        variables = {
            rewrite_key(k): v for k, v in variables.items() if not is_banned_key(k)
        }
        variables = {k: process_value(k, v) for k, v in variables.items()}
        variables = flax.traverse_util.unflatten_dict(variables, sep=".")

        model.params = variables

        return model

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
