# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test, run_op_test_with_random_inputs
from utils import Category


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.view_as_complex",
)
@pytest.mark.parametrize(
    "shape",
    [(32, 32), (64, 64)],
    ids=lambda val: f"{val}",
)
def test_complex(shape: tuple, request):
    class Complex(torch.nn.Module):
        def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
            return torch.view_as_complex(torch.stack([real, imag], dim=-1))

    run_op_test_with_random_inputs(
        Complex(),
        input_shapes=[shape, shape],
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.view_as_real",
)
@pytest.mark.parametrize(
    "shape",
    [(32, 32), (64, 64)],
    ids=lambda val: f"{val}",
)
def test_view_as_real(shape: tuple, request):
    class ViewAsReal(torch.nn.Module):
        def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
            z = torch.view_as_complex(torch.stack([real, imag], dim=-1))
            return torch.view_as_real(z)

    run_op_test_with_random_inputs(
        ViewAsReal(),
        input_shapes=[shape, shape],
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.real",
)
@pytest.mark.parametrize(
    "shape",
    [(32, 32), (64, 64)],
    ids=lambda val: f"{val}",
)
def test_real(shape: tuple, request):
    class Real(torch.nn.Module):
        def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
            z = torch.view_as_complex(torch.stack([real, imag], dim=-1))
            return torch.real(z)

    run_op_test_with_random_inputs(
        Real(),
        input_shapes=[shape, shape],
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.imag",
)
@pytest.mark.parametrize(
    "shape",
    [(32, 32), (64, 64)],
    ids=lambda val: f"{val}",
)
def test_imag(shape: tuple, request):
    class Imag(torch.nn.Module):
        def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
            z = torch.view_as_complex(torch.stack([real, imag], dim=-1))
            return torch.imag(z)

    run_op_test_with_random_inputs(
        Imag(),
        input_shapes=[shape, shape],
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.view_as_complex+torch.real+torch.imag",
)
@pytest.mark.parametrize(
    "shape",
    [(32, 32), (64, 64)],
    ids=lambda val: f"{val}",
)
def test_complex_real_imag_combined(shape: tuple, request):
    """Constructs a complex tensor, extracts real and imaginary parts, then
    reconstructs it via view_as_complex to verify round-trip correctness."""

    class ComplexRealImagCombined(torch.nn.Module):
        def forward(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
            z = torch.view_as_complex(torch.stack([real, imag], dim=-1))
            r = torch.real(z)
            i = torch.imag(z)
            return torch.view_as_complex(torch.stack([r, i], dim=-1))

    run_op_test_with_random_inputs(
        ComplexRealImagCombined(),
        input_shapes=[shape, shape],
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.add"
)
@pytest.mark.parametrize(
    "shape",
    [(32, 32)],
    ids=lambda val: f"{val}",
)
def test_complex_add(shape: tuple, request):
    class ComplexAdd(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            return x + y

    run_op_test(
        ComplexAdd(),
        [torch.randn(shape, dtype=torch.complex64), torch.randn(shape, dtype=torch.complex64)],
        framework=Framework.TORCH,
        request=request,
    )

@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.add"
)
@pytest.mark.parametrize(
    "shape",
    [(32, 32)],
    ids=lambda val: f"{val}",
)
def test_complex_add_real_imag(shape: tuple, request):
    class ComplexAdd(torch.nn.Module):
        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            real = torch.real(x) + torch.real(y)
            imag = torch.imag(x) + torch.imag(y)
            stacked = torch.stack([real, imag], dim=-1)
            return torch.view_as_complex(stacked)

    run_op_test(
        ComplexAdd(),
        [torch.randn(shape, dtype=torch.complex64), torch.randn(shape, dtype=torch.complex64)],
        framework=Framework.TORCH,
        request=request,
    )

@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="apply_rotary_emb"
)
@pytest.mark.parametrize(
    "batch_size, seq_len, n_heads, dim",
    [(2, 16, 4, 64)],
    ids=lambda val: f"{val}",
)
def test_deepseek_complex_rotary_emb(batch_size, seq_len, n_heads, dim, request):
    class ApplyRotaryEmb(torch.nn.Module):
        def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            dtype = x.dtype
            shape = x.shape
            x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
            freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
            y = torch.view_as_real(x * freqs_cis).flatten(3)
            return y.to(dtype)

    head_dim = dim // n_heads

    run_op_test(
        ApplyRotaryEmb(),
        [
            torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=torch.bfloat16),
            torch.randn(seq_len, head_dim // 2, dtype=torch.complex64),
        ],
        framework=Framework.TORCH,
        request=request,
    )