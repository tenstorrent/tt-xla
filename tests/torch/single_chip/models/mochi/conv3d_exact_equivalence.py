# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Conv3D Exact Equivalence using Im2col Approach

This implements Conv3D using Unfold + Linear operations, which is:
1. Mathematically EXACT (not an approximation)
2. Uses same number of parameters
3. Can directly load pretrained Conv3D weights
4. May be supported by TT backend (uses only unfold + matmul)

Based on im2col technique used internally by many deep learning frameworks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv3dAsLinear(nn.Module):
    """
    Exact equivalent of Conv3D using Unfold + Linear approach.

    Conv3D can be viewed as:
    1. Extract all 3D patches (neighborhoods) from input
    2. Flatten each patch
    3. Apply linear transformation to each flattened patch
    4. Reshape back to output volume

    This is EXACTLY what Conv3D does, just implemented differently!
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True
    ):
        super().__init__()

        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Handle kernel_size as int or tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if isinstance(stride, int):
            self.stride = (stride, stride, stride)
        else:
            self.stride = stride

        if isinstance(padding, int):
            self.padding = (padding, padding, padding)
        else:
            self.padding = padding

        kt, kh, kw = self.kernel_size

        # Linear layer that processes flattened patches
        # Input: flattened patch of size (C_in * kt * kh * kw)
        # Output: C_out features
        self.linear = nn.Linear(in_channels * kt * kh * kw, out_channels, bias=bias)

    def unfold_3d(self, x):
        """
        Extract 3D patches from input tensor.

        Similar to torch.nn.Unfold but for 3D (temporal-spatial).

        Args:
            x: [B, C, T, H, W]

        Returns:
            patches: [B, C*kt*kh*kw, num_patches]
            where num_patches = T_out * H_out * W_out
        """
        B, C, T, H, W = x.shape
        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride
        pt, ph, pw = self.padding

        # Apply padding if needed
        if any(p > 0 for p in self.padding):
            x = F.pad(x, (pw, pw, ph, ph, pt, pt))
            T = T + 2 * pt
            H = H + 2 * ph
            W = W + 2 * pw

        # Calculate output dimensions
        T_out = (T - kt) // st + 1
        H_out = (H - kh) // sh + 1
        W_out = (W - kw) // sw + 1

        # Extract patches using unfold (spatial first, then temporal)
        # Step 1: Unfold spatial dimensions (H, W)
        # x: [B, C, T, H, W] -> reshape to [B*C*T, 1, H, W]
        x_reshaped = x.reshape(B * C * T, 1, H, W)

        # Unfold spatial: [B*C*T, 1, H, W] -> [B*C*T, kh*kw, H_out*W_out]
        x_spatial = F.unfold(x_reshaped, kernel_size=(kh, kw), stride=(sh, sw))

        # Reshape: [B*C*T, kh*kw, H_out*W_out] -> [B, C, T, kh*kw, H_out*W_out]
        x_spatial = x_spatial.reshape(B, C, T, kh * kw, H_out * W_out)

        # Step 2: Unfold temporal dimension
        # Rearrange: [B, C, T, kh*kw, H_out*W_out] -> [B, C, kh*kw, H_out*W_out, T]
        x_spatial = x_spatial.permute(0, 1, 3, 4, 2).contiguous()

        # Reshape: [B, C, kh*kw, H_out*W_out, T] -> [B*C*kh*kw*H_out*W_out, 1, T]
        x_spatial = x_spatial.reshape(B * C * kh * kw * H_out * W_out, 1, T)

        # Unfold temporal: [B*C*kh*kw*H_out*W_out, 1, T] -> [B*C*kh*kw*H_out*W_out, kt, T_out]
        x_temporal = F.unfold(
            x_spatial.unsqueeze(-1), kernel_size=(kt, 1), stride=(st, 1)
        )
        x_temporal = x_temporal.squeeze(-1)  # Remove the extra spatial dim

        # Reshape back: [B*C*kh*kw*H_out*W_out, kt, T_out] -> [B, C, kh, kw, H_out, W_out, kt, T_out]
        x_temporal = x_temporal.reshape(B, C, kh * kw, H_out * W_out, kt, T_out)
        x_temporal = x_temporal.reshape(B, C, kh, kw, H_out, W_out, kt, T_out)

        # Rearrange to: [B, C, kt, kh, kw, T_out, H_out, W_out]
        x_temporal = x_temporal.permute(0, 1, 6, 2, 3, 7, 4, 5).contiguous()

        # Flatten patch dimensions: [B, C*kt*kh*kw, T_out*H_out*W_out]
        patches = x_temporal.reshape(B, C * kt * kh * kw, T_out * H_out * W_out)

        return patches, (T_out, H_out, W_out)

    def forward(self, x):
        """
        Args:
            x: [B, C_in, T, H, W]

        Returns:
            [B, C_out, T_out, H_out, W_out]
        """
        B = x.shape[0]

        # Step 1: Extract 3D patches (im2col)
        # patches: [B, C_in*kt*kh*kw, num_patches]
        patches, (T_out, H_out, W_out) = self.unfold_3d(x)

        # Step 2: Transpose for linear layer
        # [B, C_in*kt*kh*kw, num_patches] -> [B, num_patches, C_in*kt*kh*kw]
        patches = patches.transpose(1, 2)

        # Step 3: Apply linear transformation (this is the convolution!)
        # [B, num_patches, C_in*kt*kh*kw] -> [B, num_patches, C_out]
        output = self.linear(patches)

        # Step 4: Reshape to output volume
        # [B, num_patches, C_out] -> [B, C_out, T_out, H_out, W_out]
        output = output.transpose(1, 2)  # [B, C_out, num_patches]
        output = output.reshape(B, self.out_channels, T_out, H_out, W_out)

        return output

    @staticmethod
    def from_conv3d(conv3d):
        """
        Convert a pretrained Conv3D to Conv3dAsLinear with exact same weights.

        Args:
            conv3d: nn.Conv3d module

        Returns:
            Conv3dAsLinear module with converted weights
        """
        # Create equivalent module
        conv_linear = Conv3dAsLinear(
            in_channels=conv3d.in_channels,
            out_channels=conv3d.out_channels,
            kernel_size=conv3d.kernel_size,
            stride=conv3d.stride,
            padding=conv3d.padding,
            bias=conv3d.bias is not None,
        )

        # Convert weights
        # Conv3D weight: [C_out, C_in, kt, kh, kw]
        # Linear weight: [C_out, C_in*kt*kh*kw]
        conv3d_weight = conv3d.weight.data
        C_out, C_in, kt, kh, kw = conv3d_weight.shape

        # Flatten spatial-temporal dimensions
        linear_weight = conv3d_weight.reshape(C_out, C_in * kt * kh * kw)
        conv_linear.linear.weight.data = linear_weight

        # Copy bias if present
        if conv3d.bias is not None:
            conv_linear.linear.bias.data = conv3d.bias.data

        return conv_linear


def test_equivalence():
    """
    Test that Conv3dAsLinear produces EXACTLY the same output as Conv3D.
    """
    print("=" * 70)
    print("Testing Conv3D <-> Conv3dAsLinear Exact Equivalence")
    print("=" * 70)

    # Test parameters
    B, C_in, C_out = 2, 64, 128
    T, H, W = 8, 32, 32

    # Create input
    torch.manual_seed(42)
    x = torch.randn(B, C_in, T, H, W)

    print(f"\nInput shape: {x.shape}")

    # Test case 1: kernel=3, padding=0
    print("\n" + "-" * 70)
    print("Test 1: kernel=(3,3,3), stride=1, padding=0")
    print("-" * 70)

    conv3d = nn.Conv3d(C_in, C_out, kernel_size=3, stride=1, padding=0)
    conv_linear = Conv3dAsLinear.from_conv3d(conv3d)

    with torch.no_grad():
        y_conv3d = conv3d(x)
        y_linear = conv_linear(x)

    print(f"Conv3D output shape:    {y_conv3d.shape}")
    print(f"Linear output shape:    {y_linear.shape}")
    print(f"Max absolute difference: {(y_conv3d - y_linear).abs().max().item():.2e}")
    print(f"Mean absolute difference: {(y_conv3d - y_linear).abs().mean().item():.2e}")

    assert torch.allclose(y_conv3d, y_linear, atol=1e-5), "Outputs don't match!"
    print("✓ Outputs are numerically identical!")

    # Test case 2: kernel=3, padding=1
    print("\n" + "-" * 70)
    print("Test 2: kernel=(3,3,3), stride=1, padding=1")
    print("-" * 70)

    conv3d_p1 = nn.Conv3d(C_in, C_out, kernel_size=3, stride=1, padding=1)
    conv_linear_p1 = Conv3dAsLinear.from_conv3d(conv3d_p1)

    with torch.no_grad():
        y_conv3d_p1 = conv3d_p1(x)
        y_linear_p1 = conv_linear_p1(x)

    print(f"Conv3D output shape:    {y_conv3d_p1.shape}")
    print(f"Linear output shape:    {y_linear_p1.shape}")
    print(
        f"Max absolute difference: {(y_conv3d_p1 - y_linear_p1).abs().max().item():.2e}"
    )
    print(
        f"Mean absolute difference: {(y_conv3d_p1 - y_linear_p1).abs().mean().item():.2e}"
    )

    assert torch.allclose(y_conv3d_p1, y_linear_p1, atol=1e-5), "Outputs don't match!"
    print("✓ Outputs are numerically identical!")

    # Test case 3: kernel=1 (pointwise)
    print("\n" + "-" * 70)
    print("Test 3: kernel=(1,1,1), stride=1, padding=0 (pointwise)")
    print("-" * 70)

    conv3d_pw = nn.Conv3d(C_in, C_out, kernel_size=1, stride=1, padding=0)
    conv_linear_pw = Conv3dAsLinear.from_conv3d(conv3d_pw)

    with torch.no_grad():
        y_conv3d_pw = conv3d_pw(x)
        y_linear_pw = conv_linear_pw(x)

    print(f"Conv3D output shape:    {y_conv3d_pw.shape}")
    print(f"Linear output shape:    {y_linear_pw.shape}")
    print(
        f"Max absolute difference: {(y_conv3d_pw - y_linear_pw).abs().max().item():.2e}"
    )
    print(
        f"Mean absolute difference: {(y_conv3d_pw - y_linear_pw).abs().mean().item():.2e}"
    )

    assert torch.allclose(y_conv3d_pw, y_linear_pw, atol=1e-5), "Outputs don't match!"
    print("✓ Outputs are numerically identical!")

    # Parameter count comparison
    print("\n" + "-" * 70)
    print("Parameter Count Comparison")
    print("-" * 70)

    conv3d_params = sum(p.numel() for p in conv3d.parameters())
    linear_params = sum(p.numel() for p in conv_linear.parameters())

    print(f"Conv3D parameters:      {conv3d_params:,}")
    print(f"Conv3dAsLinear parameters: {linear_params:,}")
    print(f"Difference:             {abs(conv3d_params - linear_params):,}")

    assert conv3d_params == linear_params, "Parameter counts don't match!"
    print("✓ Parameter counts are identical!")

    print("\n" + "=" * 70)
    print("All tests passed! Conv3dAsLinear is EXACTLY equivalent to Conv3D.")
    print("=" * 70)


def visualize_im2col():
    """
    Visualize how im2col works with a simple 1D example.
    """
    print("\n" + "=" * 70)
    print("Visualizing Im2col Concept")
    print("=" * 70)

    print(
        """
Im2col (image-to-column) converts convolution into matrix multiplication:

Example: 1D convolution with kernel_size=3

Input:    [1, 2, 3, 4, 5, 6]
Kernel:   [a, b, c]

Traditional convolution:
  Position 0: 1*a + 2*b + 3*c
  Position 1: 2*a + 3*b + 4*c
  Position 2: 3*a + 4*b + 5*c
  Position 3: 4*a + 5*b + 6*c

Im2col approach:
  1. Extract patches (unfold):
     [[1, 2, 3],
      [2, 3, 4],
      [3, 4, 5],
      [4, 5, 6]]

  2. Flatten kernel:
     [a, b, c]

  3. Matrix multiply:
     [[1, 2, 3],     [a]     [1*a + 2*b + 3*c]
      [2, 3, 4],  ×  [b]  =  [2*a + 3*b + 4*c]
      [3, 4, 5],     [c]     [3*a + 4*b + 5*c]
      [4, 5, 6]]             [4*a + 5*b + 6*c]

Same result, but now it's just a matrix multiplication!

For Conv3D:
  - Each patch is a 3D cube (e.g., 3×3×3 = 27 values)
  - Flattened patch becomes a 27-dimensional vector
  - Linear layer weight is the flattened 3D kernel
  - Matrix multiply gives the same result as Conv3D

This is EXACT, not an approximation!
The number of parameters is IDENTICAL!
You can directly load pretrained Conv3D weights!
    """
    )


if __name__ == "__main__":
    # Visualize concept
    visualize_im2col()

    # Run equivalence tests
    test_equivalence()

    print("\n" + "=" * 70)
    print("Ready to test on TT backend!")
    print("=" * 70)
