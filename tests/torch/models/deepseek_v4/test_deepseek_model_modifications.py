# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import importlib.util
import math
import sys
import types
from pathlib import Path

import pytest
import torch

from third_party.tt_forge_models.deepseek_v4.modified_model import (
    model as modified_model,
)
from third_party.tt_forge_models.deepseek_v4.modified_model.kernel import (
    hc_split_sinkhorn as eager_hc_split_sinkhorn,
)
from third_party.tt_forge_models.deepseek_v4.modified_model.kernel import (
    sparse_attn as eager_sparse_attn,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
ORIGINAL_MODEL_PATH = (
    REPO_ROOT / "third_party/tt_forge_models/deepseek_v4/original_model/model.py"
)


COMMON_ARGS = dict(
    max_batch_size=2,
    max_seq_len=16,
    vocab_size=128,
    dim=64,
    moe_inter_dim=32,
    n_layers=2,
    n_mtp_layers=1,
    n_heads=4,
    q_lora_rank=32,
    head_dim=16,
    rope_head_dim=8,
    n_routed_experts=4,
    n_activated_experts=2,
    n_shared_experts=1,
    o_groups=2,
    o_lora_rank=16,
    window_size=8,
    compress_ratios=(0, 4, 0),
    index_n_heads=4,
    index_head_dim=8,
    index_topk=4,
    hc_mult=2,
    hc_sinkhorn_iters=3,
    dtype="bf16",
    scale_fmt=None,
    expert_dtype=None,
    scale_dtype="fp32",
    n_hash_layers=0,
)


def _identity_act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt=None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    if inplace:
        return x
    scale_shape = (*x.shape[:-1], math.ceil(x.shape[-1] / block_size))
    scale = torch.ones(scale_shape, device=x.device, dtype=torch.float32)
    return x, scale


def _identity_fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
):
    if inplace:
        return x
    scale_shape = (*x.shape[:-1], math.ceil(x.shape[-1] / block_size))
    scale = torch.ones(scale_shape, device=x.device, dtype=torch.float32)
    return x, scale


def _dense_gemm(
    a: torch.Tensor,
    a_s: torch.Tensor,
    b: torch.Tensor,
    b_s: torch.Tensor,
    scale_dtype: torch.dtype = torch.float32,
):
    return a.float() @ b.float().transpose(-1, -2)


@pytest.fixture(scope="session")
def original_model_module():
    kernel_stub = types.ModuleType("kernel")
    kernel_stub.act_quant = _identity_act_quant
    kernel_stub.fp4_act_quant = _identity_fp4_act_quant
    kernel_stub.fp8_gemm = _dense_gemm
    kernel_stub.fp4_gemm = _dense_gemm
    kernel_stub.sparse_attn = eager_sparse_attn
    kernel_stub.hc_split_sinkhorn = eager_hc_split_sinkhorn

    old_kernel = sys.modules.get("kernel")
    sys.modules["kernel"] = kernel_stub
    try:
        spec = importlib.util.spec_from_file_location(
            "deepseek_v4_original_model_stubbed", ORIGINAL_MODEL_PATH
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
    finally:
        if old_kernel is None:
            del sys.modules["kernel"]
        else:
            sys.modules["kernel"] = old_kernel

    module.linear = modified_model.linear
    module.rotate_activation = lambda x: x
    return module


def _named_tensors(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    tensors = dict(module.named_parameters())
    tensors.update(module.named_buffers())
    return tensors


def _sync_common_state(
    original_module: torch.nn.Module, modified_module_: torch.nn.Module
) -> None:
    orig_tensors = _named_tensors(original_module)
    mod_tensors = _named_tensors(modified_module_)

    with torch.no_grad():
        for name, orig_tensor in orig_tensors.items():
            if name not in mod_tensors:
                continue
            mod_tensor = mod_tensors[name]
            if orig_tensor.shape == mod_tensor.shape:
                mod_tensor.copy_(orig_tensor.to(dtype=mod_tensor.dtype))
                continue
            if (
                name.endswith("freqs_cis")
                and orig_tensor.dtype.is_complex
                and mod_tensor.shape == orig_tensor.shape + (2,)
            ):
                mod_tensor.copy_(torch.view_as_real(orig_tensor).to(mod_tensor.dtype))


def _prepare_original_model_state(model: torch.nn.Module) -> None:
    for layer in model.layers:
        attn = layer.attn
        if not attn.compress_ratio:
            continue
        attn.kv_cache = attn.kv_cache.to(torch.bfloat16)
        attn.compressor.kv_cache = attn.kv_cache[:, attn.window_size :]
        attn.compressor.freqs_cis = attn.freqs_cis
        if attn.indexer is not None:
            attn.indexer.kv_cache = attn.indexer.kv_cache.to(torch.bfloat16)
            attn.indexer.freqs_cis = attn.freqs_cis
            attn.indexer.compressor.kv_cache = attn.indexer.kv_cache
            attn.indexer.compressor.freqs_cis = attn.freqs_cis


def _initialize_original_parameters(module: torch.nn.Module) -> None:
    generator = torch.Generator().manual_seed(0)

    with torch.no_grad():
        for name, param in module.named_parameters():
            if param.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                values = torch.arange(param.numel(), dtype=torch.int64).reshape(
                    param.shape
                )
                param.copy_((values % COMMON_ARGS["n_routed_experts"]).to(param.dtype))
                continue

            if "norm.weight" in name:
                param.fill_(1)
                continue

            if "scale" in name:
                values = 0.1 + 0.01 * torch.arange(param.numel(), dtype=torch.float32)
                param.copy_(values.reshape(param.shape).to(param.dtype))
                continue

            if name.endswith("bias") or name.endswith("base"):
                values = torch.linspace(-0.02, 0.02, param.numel(), dtype=torch.float32)
                param.copy_(values.reshape(param.shape).to(param.dtype))
                continue

            values = 0.05 * torch.randn(
                param.shape, generator=generator, dtype=torch.float32
            )
            param.copy_(values.to(param.dtype))


def _make_args(model_module, **overrides):
    kwargs = dict(COMMON_ARGS)
    kwargs.update(overrides)
    return model_module.ModelArgs(**kwargs)


def _make_model_pair(original_model_module, **overrides):
    torch.manual_seed(0)
    original_args = _make_args(original_model_module, **overrides)
    original_model = original_model_module.Transformer(original_args).eval()
    original_model.embed.weight.data = original_model.embed.weight.data.to(
        torch.bfloat16
    )
    _initialize_original_parameters(original_model)

    torch.manual_seed(1)
    modified_args = _make_args(modified_model, **overrides)
    modified = modified_model.Transformer(modified_args).eval()
    modified.embed.weight.data = modified.embed.weight.data.to(torch.bfloat16)

    _sync_common_state(original_model, modified)
    _prepare_original_model_state(original_model)
    return original_model, modified


def _clone_args(args):
    cloned = []
    for arg in args:
        if torch.is_tensor(arg):
            cloned.append(arg.clone())
        else:
            cloned.append(arg)
    return cloned


def _assert_close(actual, expected, *, atol=1e-4, rtol=1e-4):
    if torch.is_tensor(actual):
        if actual.dtype in (
            torch.int32,
            torch.int64,
            torch.int16,
            torch.int8,
            torch.bool,
        ):
            assert torch.equal(actual, expected)
        else:
            torch.testing.assert_close(
                actual.float(), expected.float(), atol=atol, rtol=rtol
            )
        return
    assert isinstance(actual, (tuple, list))
    assert isinstance(expected, type(actual))
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        _assert_close(a, e, atol=atol, rtol=rtol)


def _get_submodule(module: torch.nn.Module, path: str) -> torch.nn.Module:
    return module.get_submodule(path)


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("ndim", [3, 4])
def test_apply_rotary_emb_matches_original(original_model_module, inverse, ndim):
    original_freqs = original_model_module.precompute_freqs_cis(
        8, 8, 0, 10000.0, 40, 32, 1
    )
    modified_freqs = modified_model.precompute_freqs_cis(8, 8, 0, 10000.0, 40, 32, 1)

    shape = (2, 8, 8) if ndim == 3 else (2, 8, 4, 8)
    x = torch.randn(shape, dtype=torch.bfloat16)
    original_x = x.clone()
    modified_x = x.clone()

    original_model_module.apply_rotary_emb(original_x, original_freqs, inverse)
    modified_model.apply_rotary_emb(modified_x, modified_freqs, inverse)
    _assert_close(modified_x, original_x, atol=5e-3, rtol=5e-3)


def test_precompute_freqs_matches_original(original_model_module):
    original_freqs = original_model_module.precompute_freqs_cis(
        8, 8, 0, 10000.0, 40, 32, 1
    )
    modified_freqs = modified_model.precompute_freqs_cis(8, 8, 0, 10000.0, 40, 32, 1)
    _assert_close(
        modified_freqs, torch.view_as_real(original_freqs), atol=1e-6, rtol=1e-6
    )


def test_topk_helpers_match_original(original_model_module):
    for start_pos in (0, 3, 8):
        original = original_model_module.get_window_topk_idxs(8, 2, 8, start_pos)
        modified = modified_model.get_window_topk_idxs(
            8, 2, 8, start_pos, torch.device("cpu")
        )
        _assert_close(modified, original)

    for start_pos in (0, 3):
        original = original_model_module.get_compress_topk_idxs(4, 2, 8, start_pos, 8)
        modified = modified_model.get_compress_topk_idxs(
            4, 2, 8, start_pos, 8, torch.device("cpu")
        )
        _assert_close(modified, original)


@pytest.mark.parametrize(
    "selector,input_builder",
    [
        ("embed", lambda model: [torch.randint(0, 128, (2, 8), dtype=torch.long)]),
        (
            "layers.0.attn.wq_a",
            lambda model: [torch.randn(2, 8, 64, dtype=torch.bfloat16)],
        ),
        (
            "layers.0.attn.wq_b",
            lambda model: [torch.randn(2, 8, 32, dtype=torch.bfloat16)],
        ),
        (
            "layers.0.attn.wo_b",
            lambda model: [torch.randn(2, 8, 32, dtype=torch.bfloat16)],
        ),
        (
            "layers.0.attn_norm",
            lambda model: [torch.randn(2, 8, 64, dtype=torch.bfloat16)],
        ),
    ],
)
def test_basic_modules_match_original(original_model_module, selector, input_builder):
    original_model, modified = _make_model_pair(original_model_module)
    original_submodule = _get_submodule(original_model, selector)
    modified_submodule = _get_submodule(modified, selector)

    inputs = input_builder(modified)
    with torch.inference_mode():
        original_out = original_submodule(*_clone_args(inputs))
        modified_out = modified_submodule(*_clone_args(inputs))
    _assert_close(modified_out, original_out, atol=5e-3, rtol=5e-3)


def test_gate_matches_original(original_model_module):
    original_model, modified = _make_model_pair(original_model_module)
    x = torch.randn(8, 64, dtype=torch.bfloat16)
    input_ids = torch.randint(0, 128, (8,), dtype=torch.long)

    with torch.inference_mode():
        original_out = original_model.layers[0].ffn.gate(x, input_ids)
        modified_out = modified.layers[0].ffn.gate(x, input_ids)
    _assert_close(modified_out, original_out, atol=1e-4, rtol=1e-4)


def test_expert_matches_original(original_model_module):
    original_model, modified = _make_model_pair(original_model_module)
    x = torch.randn(8, 64, dtype=torch.bfloat16)
    weights = torch.rand(8, 1, dtype=torch.float32)

    with torch.inference_mode():
        original_out = original_model.layers[0].ffn.experts[0](x, weights)
        modified_out = modified.layers[0].ffn.experts[0](x, weights)
    _assert_close(modified_out, original_out, atol=5e-3, rtol=5e-3)


def test_moe_matches_original(original_model_module):
    original_model, modified = _make_model_pair(original_model_module)
    x = torch.randn(2, 8, 64, dtype=torch.bfloat16)
    input_ids = torch.randint(0, 128, (2, 8), dtype=torch.long)

    with torch.inference_mode():
        original_out = original_model.layers[0].ffn(x, input_ids)
        modified_out = modified.layers[0].ffn(x, input_ids)
    _assert_close(modified_out, original_out, atol=5e-3, rtol=5e-3)


def test_parallel_head_matches_original(original_model_module):
    original_model, modified = _make_model_pair(original_model_module)
    x = torch.randn(2, 8, 2, 64, dtype=torch.bfloat16)

    with torch.inference_mode():
        original_out = original_model.head(
            x,
            original_model.hc_head_fn,
            original_model.hc_head_scale,
            original_model.hc_head_base,
            original_model.norm,
        )
        modified_out = modified.head(
            x,
            modified.hc_head_fn,
            modified.hc_head_scale,
            modified.hc_head_base,
            modified.norm,
        )
    _assert_close(modified_out, original_out, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("start_pos,seqlen", [(0, 8), (3, 1)])
def test_compressor_matches_original(original_model_module, start_pos, seqlen):
    original_model, modified = _make_model_pair(original_model_module)
    original_attn = original_model.layers[1].attn
    modified_attn = modified.layers[1].attn

    x = torch.randn(1, seqlen, 64, dtype=torch.bfloat16)
    with torch.inference_mode():
        original_out = original_attn.compressor(*_clone_args([x, start_pos]))
        modified_out = modified_attn.compressor(*_clone_args([x, start_pos]))

    _assert_close(modified_out, original_out, atol=5e-3, rtol=5e-3)
    _assert_close(
        modified_attn.compressor.kv_cache,
        original_attn.compressor.kv_cache,
        atol=5e-3,
        rtol=5e-3,
    )
    _assert_close(
        modified_attn.compressor.kv_state,
        original_attn.compressor.kv_state,
        atol=5e-3,
        rtol=5e-3,
    )
    _assert_close(
        modified_attn.compressor.score_state,
        original_attn.compressor.score_state,
        atol=5e-3,
        rtol=5e-3,
    )


@pytest.mark.parametrize("start_pos,seqlen", [(0, 8), (3, 1)])
def test_indexer_matches_original(original_model_module, start_pos, seqlen):
    original_model, modified = _make_model_pair(original_model_module)
    original_indexer = original_model.layers[1].attn.indexer
    modified_indexer = modified.layers[1].attn.indexer

    x = torch.randn(1, seqlen, 64, dtype=torch.bfloat16)
    qr = torch.randn(1, seqlen, 32, dtype=torch.bfloat16)
    offset = 8
    with torch.inference_mode():
        original_out = original_indexer(*_clone_args([x, qr, start_pos, offset]))
        modified_out = modified_indexer(*_clone_args([x, qr, start_pos, offset]))

    _assert_close(modified_out, original_out)
    _assert_close(
        modified_indexer.compressor.kv_cache,
        original_indexer.kv_cache,
        atol=5e-3,
        rtol=5e-3,
    )


@pytest.mark.parametrize(
    "selector,start_pos,seqlen",
    [
        ("layers.0.attn", 0, 8),
        ("layers.0.attn", 3, 1),
        ("layers.1.attn", 0, 8),
        ("layers.1.attn", 3, 1),
    ],
)
def test_attention_matches_original(original_model_module, selector, start_pos, seqlen):
    original_model, modified = _make_model_pair(original_model_module)
    original_attn = _get_submodule(original_model, selector)
    modified_attn = _get_submodule(modified, selector)

    x = torch.randn(1, seqlen, 64, dtype=torch.bfloat16)
    with torch.inference_mode():
        original_out = original_attn(*_clone_args([x, start_pos]))
        modified_out = modified_attn(*_clone_args([x, start_pos]))

    _assert_close(modified_out, original_out, atol=8e-3, rtol=8e-3)


def test_block_matches_original(original_model_module):
    original_model, modified = _make_model_pair(original_model_module)
    x = torch.randn(1, 8, 2, 64, dtype=torch.bfloat16)
    input_ids = torch.randint(0, 128, (1, 8), dtype=torch.long)

    with torch.inference_mode():
        original_out = original_model.layers[1](*_clone_args([x, 0, input_ids]))
        modified_out = modified.layers[1](*_clone_args([x, 0, input_ids]))
    _assert_close(modified_out, original_out, atol=1e-2, rtol=1e-2)


def test_mtp_block_matches_original(original_model_module):
    original_model, modified = _make_model_pair(original_model_module)
    x = torch.randn(1, 8, 2, 64, dtype=torch.bfloat16)
    input_ids = torch.randint(0, 128, (1, 8), dtype=torch.long)

    with torch.inference_mode():
        original_out = original_model.mtp[0](*_clone_args([x, 0, input_ids]))
        modified_out = modified.mtp[0](*_clone_args([x, 0, input_ids]))
    _assert_close(modified_out, original_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("start_pos,seq_len", [(0, 8), (3, 1)])
def test_transformer_matches_original(original_model_module, start_pos, seq_len):
    original_model, modified = _make_model_pair(original_model_module)
    input_ids = torch.randint(0, 128, (1, seq_len), dtype=torch.long)

    with torch.inference_mode():
        original_out = original_model(*_clone_args([input_ids, start_pos]))
        modified_out = modified(*_clone_args([input_ids, start_pos]))
    _assert_close(modified_out, original_out, atol=1e-2, rtol=1e-2)
