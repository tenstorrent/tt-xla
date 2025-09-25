# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn.functional as F
import pytest
from python_package.tt_torch.composite_ops import (
    gelu_composite,
    gelu_composite_with_grad,
    GELUComposite,
    enable_gelu_composite,
    disable_gelu_composite,
)


class TestGELUComposite:
    """Test suite for GELU composite operations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        # Store original gelu for restoration
        self.original_gelu = F.gelu
        yield
        # Ensure we restore the original after each test
        F.gelu = self.original_gelu

    def test_gelu_composite_forward(self):
        """Test forward pass of GELU composite."""
        # Create test tensor
        x = torch.randn(2, 4, 8)

        # Test with tanh approximation
        y_composite = gelu_composite(x, approximate="tanh")
        y_expected = F.gelu(x, approximate="tanh")

        # Check shapes match
        assert y_composite.shape == y_expected.shape

        # Check values are close (allowing for numerical differences)
        torch.testing.assert_close(y_composite, y_expected, rtol=1e-5, atol=1e-5)

        # Test with no approximation
        y_composite_exact = gelu_composite(x, approximate="none")
        y_expected_exact = F.gelu(x, approximate="none")

        torch.testing.assert_close(
            y_composite_exact, y_expected_exact, rtol=1e-5, atol=1e-5
        )

    def test_gelu_composite_with_grad_backward(self):
        """Test backward pass of GELU composite with gradients."""
        # Create test tensor with gradient tracking
        x = torch.randn(2, 4, 8, requires_grad=True)

        # Compute with composite
        y_composite = gelu_composite_with_grad(x, approximate="tanh")
        loss_composite = y_composite.sum()
        loss_composite.backward()
        grad_composite = x.grad.clone()

        # Reset gradients
        x.grad.zero_()

        # Compute with standard GELU
        y_standard = F.gelu(x, approximate="tanh")
        loss_standard = y_standard.sum()
        loss_standard.backward()
        grad_standard = x.grad.clone()

        # Check gradients are close
        torch.testing.assert_close(grad_composite, grad_standard, rtol=1e-4, atol=1e-4)

    def test_gelu_module(self):
        """Test GELUComposite module."""
        # Create module
        gelu_module = GELUComposite(approximate="tanh")

        # Test forward
        x = torch.randn(2, 4, 8)
        y = gelu_module(x)
        y_expected = F.gelu(x, approximate="tanh")

        torch.testing.assert_close(y, y_expected, rtol=1e-5, atol=1e-5)

        # Test repr
        assert "approximate=tanh" in repr(gelu_module)

    def test_invalid_approximation(self):
        """Test that invalid approximation raises an error."""
        x = torch.randn(2, 4)

        with pytest.raises(ValueError, match="Unsupported approximate value"):
            gelu_composite(x, approximate="invalid")

    def test_monkey_patch(self):
        """Test monkey-patching of F.gelu."""
        # Enable composite
        enable_gelu_composite()

        # Create CPU tensor (should use original)
        x_cpu = torch.randn(2, 4)
        y_cpu = F.gelu(x_cpu)

        # The patched version should still work on CPU
        y_cpu_expected = self.original_gelu(x_cpu)
        # Use more relaxed tolerance as we might be using the approximation
        torch.testing.assert_close(y_cpu, y_cpu_expected, rtol=1e-3, atol=1e-3)

        # Disable composite
        disable_gelu_composite()

        # Check it's restored
        assert F.gelu == self.original_gelu

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gelu_composite_cuda(self):
        """Test GELU composite on CUDA device if available."""
        device = torch.device("cuda")
        x = torch.randn(2, 4, 8, device=device)

        y_composite = gelu_composite(x, approximate="tanh")
        y_expected = F.gelu(x, approximate="tanh")

        torch.testing.assert_close(y_composite, y_expected, rtol=1e-5, atol=1e-5)


@pytest.mark.xfail(reason="XLA device might not be available in test environment")
class TestGELUCompositeXLA:
    """Test suite for GELU composite operations on XLA device."""

    def test_gelu_composite_xla(self):
        """Test GELU composite on XLA device."""
        import torch_xla.core.xla_model as xm

        device = xm.xla_device()
        x = torch.randn(2, 4, 8).to(device)

        # Test composite version
        y_composite = gelu_composite(x, approximate="tanh")

        # Test with gradient
        x_grad = torch.randn(2, 4, 8, requires_grad=True).to(device)
        y_grad = gelu_composite_with_grad(x_grad, approximate="tanh")
        loss = y_grad.sum()
        loss.backward()

        assert x_grad.grad is not None
        assert x_grad.grad.shape == x_grad.shape

    def test_monkey_patch_xla(self):
        """Test monkey-patched F.gelu on XLA device."""
        import torch_xla.core.xla_model as xm

        enable_gelu_composite()

        device = xm.xla_device()
        x = torch.randn(2, 4, 8).to(device)

        # This should use the composite version
        y = F.gelu(x)

        assert y.device == device
        assert y.shape == x.shape

        disable_gelu_composite()
