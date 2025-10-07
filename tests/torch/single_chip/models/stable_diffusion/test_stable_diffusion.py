# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode
from infra.comparators import ComparisonConfig, TorchComparator, PccConfig, AtolConfig
from utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    incorrect_result,
)

import torch
import torch_xla.runtime as xr
import torch_xla

from diffusers import StableDiffusion3Pipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    retrieve_timesteps,
)


@pytest.mark.parametrize("seq_len", [256])
def test_stable_diffusion_transformer_module(seq_len):
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.float32
    )
    transformer_module = pipe.transformer
    transformer_module.eval()

    # Args for forward pass based on SD3Transformer2DModel's forward method:
    """
    Args:
            hidden_states (`torch.Tensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.Tensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`):
                Embeddings projected from the embeddings of input conditions.
            timestep (`torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.
            skip_layers (`list` of `int`, *optional*):
                A list of layer indices to skip during the forward pass.
    """

    # Prepare inputs for transformer module
    batch_size = 1
    test_seq_len = seq_len
    in_channels = transformer_module.config.in_channels
    sample_size = transformer_module.config.sample_size
    embed_dims = transformer_module.config.joint_attention_dim
    projection_dim = transformer_module.config.pooled_projection_dim

    hidden_states = torch.randn(
        batch_size, in_channels, sample_size, sample_size
    )  # (batch size, channel, height, width)
    encoder_hidden_states = torch.randn(
        batch_size, test_seq_len, embed_dims
    )  # (batch size, sequence_len, embed_dims)
    pooled_projections = torch.randn(
        batch_size, projection_dim
    )  # (batch size, projection_dim)
    timestep = torch.tensor(1000)  # ()
    block_controlnet_hidden_states = None  # (list of tensors)
    joint_attention_kwargs = None  # (dict) optional
    return_dict = True  # (bool) optional but by default True
    skip_layers = None  # (list of int) optional

    inputs = {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "pooled_projections": pooled_projections,
        "timestep": timestep,
        "block_controlnet_hidden_states": block_controlnet_hidden_states,
        "joint_attention_kwargs": joint_attention_kwargs,
        "return_dict": return_dict,
        "skip_layers": skip_layers,
    }

    # Run on CPU for golden
    cpu_output = transformer_module(**inputs)

    # Compile on TT device
    xr.set_device_type("TT")
    device = torch_xla.device()
    compiled_transformer = torch.compile(transformer_module.to("TT"), backend="tt")

    # Run on TT device
    tt_output = compiled_transformer(**inputs)

    # compare outputs
    comparator = TorchComparator(
        ComparisonConfig(
            # atol=AtolConfig(required_atol=0.02),
            pcc=PccConfig(required_pcc=0.99),
        )
    )

    # comparator.compare(tt_output, cpu_output)
    print("CPU output: ", cpu_output)
    print("TT output: ", tt_output)
