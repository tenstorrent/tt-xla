# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from infra import Framework, RunMode, ComparisonConfig
from utils import (
    BringupStatus,
    Category,
)
from infra.comparators import ComparisonConfig, TorchComparator, PccConfig

import torch
import torch_xla.runtime as xr
import torch_xla

from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from third_party.tt_forge_models.stable_diffusion.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)


VARIANT_NAME = ModelVariant.STABLE_DIFFUSION_3_5_LARGE
MODEL_INFO = ModelLoader.get_model_info(VARIANT_NAME)


def sd3_5_tf(seq_len):
    print("Starting test...")
    loader = ModelLoader(variant=VARIANT_NAME)
    model = loader.load_model(dtype_override=torch.bfloat16)

    print("Loading transformer module...")
    transformer_module = model.transformer
    transformer_module.eval()
    print("Transformer module loaded successfully")

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

    # Prepare custom dummy inputs for transformer module
    print("Preparing inputs for transformer module...")
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
    timestep = torch.tensor([1000], dtype=torch.long)  # ()

    inputs = {
        "hidden_states": hidden_states,
        "encoder_hidden_states": encoder_hidden_states,
        "pooled_projections": pooled_projections,
        "timestep": timestep,
    }

    inputs = {
        k: v.to(torch.bfloat16) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    # Run on CPU for golden
    print("Running on CPU for golden...")
    cpu_output = transformer_module(**inputs)

    print("cpu_output received!: ", cpu_output)

    # Compile
    print("Compiling...")
    compiled_transformer = torch.compile(transformer_module, backend="tt")

    # set up device
    print("Setting up device...")
    xr.set_device_type("TT")
    device = torch_xla.device()

    # Move inputs to TT device
    print("Moving inputs and model to TT device...")
    compiled_transformer.to(device)
    tt_inputs = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
    }

    # Run on TT device
    print("Running on TT device...")
    tt_output = compiled_transformer(**tt_inputs)

    print("tt_output received!: ", tt_output)

    # compare outputs
    comparator = TorchComparator(
        ComparisonConfig(
            pcc=PccConfig(required_pcc=0.99),
        )
    )

    comparator.compare(tt_output, cpu_output)


# --------------------------------
# main
# --------------------------------


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.OTHER,
    model_info=MODEL_INFO,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.NOT_STARTED,
)
@pytest.mark.parametrize("seq_len", [128])
def test_sd3_5_tf(seq_len):
    try:
        xr.set_device_type("TT")
    except Exception as e:
        pytest.skip(f"TT device not available: {e}")

    results = sd3_5_tf(seq_len)
