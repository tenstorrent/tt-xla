# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Bisect Z-Image transformer SHLO→TTIR RoPE / complex legalization (issue #4756).

Requires tt-mlir branch ``akannan/zimage_shlo_bug`` for gather/index/cat slices.

  source venv/activate
  python -m pytest -svv tests/torch/models/z_image/test_transformer_slice.py -k single_chip
"""

import pytest
import torch
import torch_xla.runtime as xr
from diffusers.models.transformers.transformer_z_image import ZSingleStreamAttnProcessor
from infra import Framework, run_op_test
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.z_image.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.z_image.pytorch.src.model_utils import MESH_SHAPES

from .slice_helpers import (
    ApplyRotaryEmbModule,
    AttentionWithRopeModule,
    PrepareSequenceRopeXModule,
    RopeEmbedderModule,
    RopeIndexAndCatModule,
    RopeIndexAxisModule,
    RopePolarThenIndexAxis1Module,
    RopePolarAxisModule,
    RopePrecomputeAllAxesModule,
    fresh_rope_embedder,
    load_transformer_slice_bundle,
    shard_spec_for_block,
)

# RoPE / gather slices: pass with tt-mlir akannan/zimage_shlo_bug.
RUN_ON_TT_SLICES = frozenset(
    {
        "rope_precompute_all_axes",
        "rope_index_axis0_precomputed",
        "rope_index_axis1_precomputed",
        "rope_index_axis2_precomputed",
        "rope_index_and_cat_precomputed",
        "rope_polar_then_index_axis1",
        "rope_polar_axis0",
        "rope_polar_axis1",
        "rope_polar_axis2",
        "rope_embedder_x",
        "rope_embedder_cap",
        "prepare_sequence_rope_x",
        "apply_rotary_emb",
        "attention_x",
        "noise_refiner_0",
        "context_refiner_0",
        "main_layer_0",
    }
)

ALL_SLICES = tuple(sorted(RUN_ON_TT_SLICES))

_ROPE_NO_SHARD = frozenset(
    {
        "rope_precompute_all_axes",
        "rope_index_axis0_precomputed",
        "rope_index_axis1_precomputed",
        "rope_index_axis2_precomputed",
        "rope_index_and_cat_precomputed",
        "rope_polar_then_index_axis1",
        "rope_polar_axis0",
        "rope_polar_axis1",
        "rope_polar_axis2",
        "rope_embedder_x",
        "rope_embedder_cap",
        "prepare_sequence_rope_x",
        "apply_rotary_emb",
    }
)


@pytest.fixture(scope="module")
def bundle():
    return load_transformer_slice_bundle(dtype=torch.bfloat16)


def _build_slice(name: str, bundle):
    t = bundle.transformer
    dummy = torch.zeros(1, dtype=torch.bfloat16)
    if name == "rope_precompute_all_axes":
        return RopePrecomputeAllAxesModule(t.rope_embedder), [dummy]
    if name == "rope_index_axis0_precomputed":
        return RopeIndexAxisModule(t.rope_embedder, axis_idx=0), [bundle.x_pos_ids_cat]
    if name == "rope_index_axis1_precomputed":
        return RopeIndexAxisModule(t.rope_embedder, axis_idx=1), [bundle.x_pos_ids_cat]
    if name == "rope_index_axis2_precomputed":
        return RopeIndexAxisModule(t.rope_embedder, axis_idx=2), [bundle.x_pos_ids_cat]
    if name == "rope_index_and_cat_precomputed":
        return RopeIndexAndCatModule(t.rope_embedder), [bundle.x_pos_ids_cat]
    if name == "rope_polar_then_index_axis1":
        return RopePolarThenIndexAxis1Module(t.rope_embedder, axis_idx=1), [
            bundle.x_pos_ids_cat
        ]
    if name == "rope_polar_axis1":
        re = t.rope_embedder
        return RopePolarAxisModule(re.axes_dims, re.axes_lens, re.theta, axis_idx=1), [
            dummy
        ]
    if name == "rope_polar_axis0":
        re = t.rope_embedder
        return RopePolarAxisModule(re.axes_dims, re.axes_lens, re.theta, axis_idx=0), [
            dummy
        ]
    if name == "rope_polar_axis2":
        re = t.rope_embedder
        return RopePolarAxisModule(re.axes_dims, re.axes_lens, re.theta, axis_idx=2), [
            dummy
        ]
    if name == "rope_embedder_x":
        return RopeEmbedderModule(fresh_rope_embedder(t)), [bundle.x_pos_ids_cat]
    if name == "rope_embedder_cap":
        return RopeEmbedderModule(fresh_rope_embedder(t)), [bundle.cap_pos_ids_cat]
    if name == "prepare_sequence_rope_x":
        return PrepareSequenceRopeXModule(t), [bundle.x_pos_ids_cat]
    if name == "apply_rotary_emb":
        block = t.noise_refiner[0]
        heads = t.n_heads
        head_dim = t.dim // heads
        normed = block.attention_norm1(bundle.x_after_prepare)
        return ApplyRotaryEmbModule(), [normed, bundle.x_freqs]
    if name == "attention_x":
        block = t.noise_refiner[0]
        proc = ZSingleStreamAttnProcessor()
        normed = block.attention_norm1(bundle.x_after_prepare)
        return AttentionWithRopeModule(block.attention, proc), [normed, bundle.x_freqs]
    if name == "noise_refiner_0":
        block = t.noise_refiner[0]
        return block, [
            bundle.x_after_prepare,
            bundle.x_mask,
            bundle.x_freqs,
            bundle.adaln_input,
            None,
            None,
            None,
        ]
    if name == "context_refiner_0":
        block = t.context_refiner[0]
        return block, [bundle.cap_after_prepare, bundle.cap_mask, bundle.cap_freqs]
    if name == "main_layer_0":
        block = t.layers[0]
        return block, [
            bundle.x_after_prepare,
            bundle.x_mask,
            bundle.x_freqs,
            bundle.adaln_input,
            None,
            None,
            None,
        ]
    raise ValueError(f"unknown slice: {name}")


def _mesh_and_shard(sharded: bool, name: str, bundle):
    mesh = None
    shard_spec_fn = None
    if not sharded:
        return mesh, shard_spec_fn

    num_devices = xr.global_runtime_device_count()
    if num_devices < 2:
        pytest.skip(f"sharded slice requires >= 2 TT devices, got {num_devices}")
    if num_devices not in MESH_SHAPES:
        pytest.skip(f"unsupported device count {num_devices}")

    loader = ModelLoader(ModelVariant.TRANSFORMER)
    mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
    mesh = get_mesh(mesh_shape, mesh_names)

    if name in _ROPE_NO_SHARD:
        shard_spec_fn = None
    else:
        model, _ = _build_slice(name, bundle)
        if hasattr(model, "parameters"):
            shard_spec_fn = shard_spec_for_block(bundle.transformer, model)

    return mesh, shard_spec_fn


def _run_op(slice_name: str, bundle, *, sharded: bool):
    xr.set_device_type("TT")
    model, inputs = _build_slice(slice_name, bundle)
    model = model.eval()
    mesh, shard_spec_fn = _mesh_and_shard(sharded, slice_name, bundle)
    run_op_test(
        model,
        inputs,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
    )


@pytest.mark.model_test
@pytest.mark.parametrize("slice_name", ALL_SLICES)
def test_transformer_slice_runs_single_chip(slice_name, bundle):
    _run_op(slice_name, bundle, sharded=False)


@pytest.mark.model_test
@pytest.mark.parametrize("slice_name", ALL_SLICES)
def test_transformer_slice_runs_sharded(slice_name, bundle):
    _run_op(slice_name, bundle, sharded=True)
