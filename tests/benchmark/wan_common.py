# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Shared logic for the Wan component benchmarks (TI2V-5B and I2V-A14B).

Both families (``tests/torch/models/wan5b`` and ``.../wan14b``) expose the same
public interface from their ``shared.py`` (loaders, wrappers, ``RESOLUTIONS``,
``LATENT_CHANNELS``, ``wan22_mesh``, ``shard_*_specs``,
``apply_dit_sp_activation_sharding``) and from their ``monkey_patch.py``
(``_patch_wan_resample_rep_sentinel``, ``_patch_wan_time_embedder_dtype_probe``,
``safe_xla_slicing``, ``torch_function_override_disabled``). The per-component
benchmark bodies are therefore identical apart from a handful of family-specific
choices captured in a :class:`WanFamily` manifest — and crucially everything is
loaded **from each family's own directory** (5B from wan5b, A14B from wan14b),
not shared, so the two stay cleanly separated. ``test_wan5b.py`` /
``test_wan14b.py`` build one manifest each and delegate their (discoverable)
test functions here.
"""

import json
from contextlib import nullcontext
from dataclasses import dataclass
from types import ModuleType
from typing import Callable, Sequence

import pytest
import torch
import torch_xla.distributed.spmd as xs
from benchmarks.video_gen_benchmark import benchmark_video_gen_torch_xla
from utils import aggregate_ttnn_perf_metrics, resolve_display_name

from tests.infra.testers.compiler_config import CompilerConfig

# Shared, family-independent benchmark constants.
SEED = 42
TEXT_SEQ_LEN = 512  # UMT5 output / DiT cross-attention sequence length
TEXT_EMBED_DIM = 4096  # UMT5 hidden size
DENOISE_TIMESTEP = 500.0  # arbitrary mid-schedule step for a single forward
DIT_MAX_BLOCKS = 0  # 0 = run the full transformer (30 blocks 5B / 40 blocks A14B)

# (resolution, sharded) variants, with clean node-ids for matrix references,
# e.g. tests/benchmark/test_wan14b.py::test_wan_dit[480p-sharded]
VARIANTS = [
    pytest.param("480p", False, id="480p-unsharded"),
    pytest.param("480p", True, id="480p-sharded"),
    pytest.param("720p", False, id="720p-unsharded"),
    pytest.param("720p", True, id="720p-sharded"),
]

# Sharding-only variants for components whose output is resolution-independent
# (the UMT5 text encoder), e.g. ...::test_wan_umt5[sharded]
SHARDING_VARIANTS = [
    pytest.param(False, id="unsharded"),
    pytest.param(True, id="sharded"),
]


# ---------------------------------------------------------------------------
# DiT timestep conventions (the one behavioural difference between families)
# ---------------------------------------------------------------------------


def per_patch_timestep(shapes: dict) -> torch.Tensor:
    """5B TI2V (``expand_timesteps=True``) — one timestep per patch token."""
    num_patches = (
        shapes["latent_frames"] * (shapes["latent_h"] // 2) * (shapes["latent_w"] // 2)
    )
    return torch.full((1, num_patches), DENOISE_TIMESTEP, dtype=torch.bfloat16)


def scalar_timestep(shapes: dict) -> torch.Tensor:
    """A14B (``expand_timesteps=False``) — a single scalar timestep per batch."""
    return torch.full((1,), DENOISE_TIMESTEP, dtype=torch.bfloat16)


@dataclass(frozen=True)
class WanFamily:
    """Everything loaded from one family's directory (wan5b / wan14b).

    Attributes:
        name_prefix: Reporting prefix, e.g. "Wan2.2-TI2V-5B".
        shared: The family's ``shared.py`` module — loaders, wrappers,
            RESOLUTIONS, LATENT_CHANNELS, ``wan22_mesh``, ``shard_*_specs`` and
            ``apply_dit_sp_activation_sharding``.
        dit_patches: Monkey-patch callables applied before the DiT run
            (mirrors the family's ``test_wan_dit.py``).
        vae_decoder_patches: Monkey-patch callables applied before the VAE
            decoder run (mirrors the family's ``test_vae_decoder.py``).
        dit_timestep: ``fn(shapes) -> timestep tensor`` (per-patch vs scalar).
        safe_xla_slicing: Context manager wrapping the whole VAE decoder run
            (the family's ``monkey_patch.safe_xla_slicing``).
        dit_override_disabled: Context manager wrapping the whole DiT run — the
            family's ``monkey_patch.torch_function_override_disabled``, which
            pops the tt_torch matmul/linear override for the trace + execution.
    """

    name_prefix: str
    shared: ModuleType
    dit_patches: Sequence[Callable[[], None]]
    vae_decoder_patches: Sequence[Callable[[], None]]
    dit_timestep: Callable[[dict], torch.Tensor]
    safe_xla_slicing: Callable
    dit_override_disabled: Callable


# ---------------------------------------------------------------------------
# Sharding helpers
# ---------------------------------------------------------------------------


def _mark_specs(specs, mesh):
    """Apply ``xs.mark_sharding`` for every (tensor, partition_spec) pair."""
    for tensor, spec in specs.items():
        xs.mark_sharding(tensor, mesh, spec)


def make_sharding_fn(
    spec_fn, submodule_attr, *, mesh_aware=False, sp_activation_fn=None
):
    """Build an ``apply_sharding_fn(wrapper, mesh)`` for one Wan component.

    Args:
        spec_fn: the family's ``shard_*_specs`` function returning a
            ``{tensor: partition_spec}`` dict. Called as ``spec_fn(submodule, mesh)``
            when ``mesh_aware`` else ``spec_fn(submodule)``.
        submodule_attr: wrapper attribute holding the module to shard
            (e.g. "encoder", "vae", "dit").
        mesh_aware: whether ``spec_fn`` consumes the mesh (the VAE specs do;
            UMT5 and DiT weight specs don't).
        sp_activation_fn: optional ``fn(submodule, mesh)`` registering
            sequence-parallel activation hooks (the DiT only).
    """

    def apply_sharding(wrapper, mesh):
        submodule = getattr(wrapper, submodule_attr)
        specs = spec_fn(submodule, mesh) if mesh_aware else spec_fn(submodule)
        _mark_specs(specs, mesh)
        if sp_activation_fn is not None:
            sp_activation_fn(submodule, mesh)

    return apply_sharding


# ---------------------------------------------------------------------------
# Benchmark driver + per-component runners
# ---------------------------------------------------------------------------


def run_wan_benchmark(
    *,
    model_info_name,
    wrapper,
    inputs,
    compiler_config,
    mesh_fn,
    apply_sharding_fn,
    sharded,
    output_file,
    request,
    required_pcc=None,
    run_context=nullcontext,
    compile_context=nullcontext,
):
    """Run a Wan component benchmark and persist the JSON result."""

    display_name = resolve_display_name(request=request, fallback=model_info_name)
    ttnn_perf_metrics_output_file = f"tt_xla_{display_name}_perf_metrics"

    print(f"Running Wan benchmark: {model_info_name} (sharded={sharded})")

    results = benchmark_video_gen_torch_xla(
        wrapper=wrapper,
        inputs=inputs,
        model_info_name=model_info_name,
        display_name=display_name,
        compiler_config=compiler_config,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        sharded=sharded,
        mesh_fn=mesh_fn,
        apply_sharding_fn=apply_sharding_fn,
        run_context=run_context,
        compile_context=compile_context,
        required_pcc=required_pcc,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name
        aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results)
        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)


def _trace_compiler_config(**overrides):
    """CompilerConfig with optimization_level=1 and trace enabled."""
    return CompilerConfig(optimization_level=1, enable_trace=True, **overrides)


def benchmark_umt5(family, sharded, output_file, request):
    """UMT5-XXL text encoder. NOTE: unsharded variants OOM on a single device."""
    shared = family.shared
    torch.manual_seed(SEED)
    wrapper = shared.UMT5Wrapper(shared.load_umt5()).eval().bfloat16()

    vocab_size = wrapper.encoder.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, TEXT_SEQ_LEN), dtype=torch.long)
    attention_mask = torch.ones(1, TEXT_SEQ_LEN, dtype=torch.long)

    run_wan_benchmark(
        model_info_name=f"{family.name_prefix}-UMT5-Text-Encoder",
        wrapper=wrapper,
        inputs=[input_ids, attention_mask],
        compiler_config=_trace_compiler_config(),
        mesh_fn=family.shared.wan22_mesh,
        apply_sharding_fn=make_sharding_fn(family.shared.shard_umt5_specs, "encoder"),
        sharded=sharded,
        output_file=output_file,
        request=request,
    )


def benchmark_vae_encoder(family, resolution, sharded, output_file, request):
    """3D Causal VAE encoder: single-frame image -> latent mean."""
    shared = family.shared
    torch.manual_seed(SEED)
    shapes = shared.RESOLUTIONS[resolution]
    wrapper = shared.VAEEncoderWrapper(shared.load_vae()).eval().bfloat16()

    x = torch.randn(1, 3, 1, shapes["video_h"], shapes["video_w"], dtype=torch.bfloat16)

    run_wan_benchmark(
        model_info_name=f"{family.name_prefix}-VAE-Encoder",
        wrapper=wrapper,
        inputs=[x],
        compiler_config=_trace_compiler_config(),
        mesh_fn=family.shared.wan22_mesh,
        apply_sharding_fn=make_sharding_fn(
            family.shared.shard_vae_encoder_specs, "vae", mesh_aware=True
        ),
        sharded=sharded,
        output_file=output_file,
        request=request,
    )


def benchmark_vae_decoder(family, resolution, sharded, output_file, request):
    """3D Causal VAE decoder: latent -> reconstructed pixels.

    Needs the WanResample rep-sentinel patch and the XLA slice-bounds guard
    (``safe_xla_slicing``) wrapping the whole run, mirroring the family's
    ``test_vae_decoder.py``.
    """
    for patch in family.vae_decoder_patches:
        patch()

    shared = family.shared
    torch.manual_seed(SEED)
    shapes = shared.RESOLUTIONS[resolution]
    wrapper = shared.VAEDecoderWrapper(shared.load_vae()).eval().bfloat16()

    z = torch.randn(
        1,
        shared.LATENT_CHANNELS,
        shapes["latent_frames"],
        shapes["latent_h"],
        shapes["latent_w"],
        dtype=torch.bfloat16,
    )

    run_wan_benchmark(
        model_info_name=f"{family.name_prefix}-VAE-Decoder",
        wrapper=wrapper,
        inputs=[z],
        compiler_config=_trace_compiler_config(
            experimental_enable_dram_space_saving_optimization=True,
        ),
        mesh_fn=family.shared.wan22_mesh,
        apply_sharding_fn=make_sharding_fn(
            family.shared.shard_vae_decoder_specs, "vae", mesh_aware=True
        ),
        sharded=sharded,
        output_file=output_file,
        request=request,
        compile_context=family.safe_xla_slicing,
    )


def benchmark_dit(family, resolution, sharded, output_file, request):
    """WanDiT transformer: one denoising forward pass.

    DiT sharding is TP weight specs + SP activation hooks, and the torch
    function override is disabled across compile + execution.
    """
    for patch in family.dit_patches:
        patch()

    shared = family.shared
    torch.manual_seed(SEED)
    shapes = shared.RESOLUTIONS[resolution]
    t, h, w = shapes["latent_frames"], shapes["latent_h"], shapes["latent_w"]
    wrapper = (
        shared.WanDiTWrapper(shared.load_dit(max_blocks=DIT_MAX_BLOCKS))
        .eval()
        .bfloat16()
    )

    # DiT input channels vary by model (5B TI2V=48, A14B-I2V=36) — take them
    # from the loaded transformer's config, not the VAE z_dim (LATENT_CHANNELS),
    # so the input matches whichever expert was loaded.
    in_channels = wrapper.dit.config.in_channels
    hidden_states = torch.randn(1, in_channels, t, h, w, dtype=torch.bfloat16)
    timestep = family.dit_timestep(shapes)
    encoder_hidden_states = torch.randn(
        1, TEXT_SEQ_LEN, TEXT_EMBED_DIM, dtype=torch.bfloat16
    )

    run_wan_benchmark(
        model_info_name=f"{family.name_prefix}-DiT",
        wrapper=wrapper,
        inputs=[hidden_states, timestep, encoder_hidden_states],
        compiler_config=_trace_compiler_config(
            experimental_enable_dram_space_saving_optimization=True,
        ),
        mesh_fn=family.shared.wan22_mesh,
        apply_sharding_fn=make_sharding_fn(
            family.shared.shard_dit_specs,
            "dit",
            sp_activation_fn=family.shared.apply_dit_sp_activation_sharding,
        ),
        sharded=sharded,
        output_file=output_file,
        request=request,
        compile_context=family.dit_override_disabled,
    )
