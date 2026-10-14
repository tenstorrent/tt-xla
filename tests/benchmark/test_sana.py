# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Sana text-to-image diffusion-transformer benchmark.

Sana (``Efficient-Large-Model/Sana_1600M_1024px``) is a fast text-to-image
diffusion transformer. Its per-step cost is dominated by the
``SanaTransformer2DModel`` (DiT) forward, which is the only component the
``sana`` loader materializes. This benchmark drives that DiT through the
shared single-forward vision harness (``benchmarks/vision_benchmark.py``):

  - The latent ``hidden_states`` is the per-iteration input.
  - The text conditioning ``encoder_hidden_states`` and the ``timestep`` are
    held fixed — they are shape-invariant across denoising steps — so each
    iteration measures one steady-state DiT step.

This mirrors the ``test_vision.py`` / ``vision_benchmark.py`` split: model
config lives here, the reusable measurement + PCC logic lives in ``benchmarks/``.
"""

import json

import torch
import torch.nn as nn
from benchmarks.vision_benchmark import benchmark_vision_torch_xla
from utils import aggregate_ttnn_perf_metrics, resolve_display_name

from third_party.tt_forge_models.sana.pytorch.loader import ModelLoader, ModelVariant

# Defaults for the Sana DiT benchmark.
DEFAULT_OPTIMIZATION_LEVEL = 2
DEFAULT_TRACE_ENABLED = False
DEFAULT_BATCH_SIZE = 1
DEFAULT_LOOP_COUNT = 32
DEFAULT_DATA_FORMAT = torch.bfloat16
DEFAULT_REQUIRED_PCC = 0.97


class _SanaTransformerForward(nn.Module):
    """Single-tensor forward adapter for the vision harness.

    The vision harness calls ``model(tensor)`` with one input per iteration, but
    ``SanaTransformer2DModel.forward`` also needs the text conditioning and the
    timestep. Those are shape-invariant across denoising steps, so we hold them
    as (non-persistent) buffers and expose a ``forward(hidden_states)`` that
    returns the predicted-noise sample tensor.
    """

    def __init__(self, transformer, encoder_hidden_states, timestep):
        super().__init__()
        self.transformer = transformer
        self.register_buffer(
            "encoder_hidden_states", encoder_hidden_states, persistent=False
        )
        self.register_buffer("timestep", timestep, persistent=False)

    def forward(self, hidden_states):
        out = self.transformer(
            hidden_states,
            encoder_hidden_states=self.encoder_hidden_states,
            timestep=self.timestep,
            return_dict=False,
        )
        # SanaTransformer2DModel returns a 1-tuple of the sample when
        # return_dict=False.
        return out[0]


def test_sana(output_file, request):
    data_format = DEFAULT_DATA_FORMAT
    batch_size = DEFAULT_BATCH_SIZE

    variant = ModelVariant.SANA_1600M_1024PX
    loader = ModelLoader(variant=variant)
    model_info_name = loader.get_model_info(variant=variant).name

    transformer = loader.load_model(dtype_override=data_format).eval()

    # Fixed conditioning + timestep (shape-invariant across denoising steps).
    sample_inputs = loader.load_inputs(
        dtype_override=data_format, batch_size=batch_size
    )
    model = _SanaTransformerForward(
        transformer,
        encoder_hidden_states=sample_inputs["encoder_hidden_states"],
        timestep=sample_inputs["timestep"],
    ).eval()

    # The harness drives the DiT with the latent only; (channels, H, W) is used
    # for reporting / export naming.
    latent_size = tuple(sample_inputs["hidden_states"].shape[1:])

    def load_inputs_fn(batch_size, dtype):
        return torch.randn(batch_size, *latent_size, dtype=dtype)

    def extract_output_tensor_fn(output):
        return output

    resolved_display_name = resolve_display_name(
        request=request, fallback=model_info_name
    )
    ttnn_perf_metrics_output_file = f"tt_xla_{resolved_display_name}_perf_metrics"

    print(f"Running Sana DiT benchmark for model: {model_info_name}")
    print(
        f"""Configuration:
    optimization_level={DEFAULT_OPTIMIZATION_LEVEL}
    trace_enabled={DEFAULT_TRACE_ENABLED}
    batch_size={batch_size}
    loop_count={DEFAULT_LOOP_COUNT}
    latent_size={latent_size}
    data_format={data_format}
    required_pcc={DEFAULT_REQUIRED_PCC}
    ttnn_perf_metrics_output_file={ttnn_perf_metrics_output_file}
    """
    )

    results = benchmark_vision_torch_xla(
        model=model,
        model_info_name=model_info_name,
        display_name=resolved_display_name,
        optimization_level=DEFAULT_OPTIMIZATION_LEVEL,
        trace_enabled=DEFAULT_TRACE_ENABLED,
        batch_size=batch_size,
        loop_count=DEFAULT_LOOP_COUNT,
        input_size=latent_size,
        data_format=data_format,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        load_inputs_fn=load_inputs_fn,
        extract_output_tensor_fn=extract_output_tensor_fn,
        required_pcc=DEFAULT_REQUIRED_PCC,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name

        aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results)

        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)
