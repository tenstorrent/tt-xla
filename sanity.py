# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn

ALIGNMENT = 32  # valid L1 alignment on Wormhole / Blackhole

# --- Qwen3.5-27B VLM patch-embed conv3d shape (from the dumped TTIR) ---
INPUT_SHAPE = (8, 3, 2, 16, 16)          # (N patches, C_in, D, H, W); CB size is N-independent
OUT_CHANNELS = 1152
KERNEL = (2, 16, 16)
STRIDE = (2, 16, 16)
PADDING = (0, 0, 0)
GROUPS = 1


def _out_size(dim, pad, stride, kernel, dilation=1):
    return (dim + 2 * pad - dilation * (kernel - 1) - 1) // stride + 1


def _prepare_input(input_tensor, C, device):
    """Permute (N,C,D,H,W) -> (N,D,H,W,C), pad C to ALIGNMENT, ROW_MAJOR on device."""
    x = input_tensor.permute(0, 2, 3, 4, 1)
    if C % ALIGNMENT != 0:
        x = torch.nn.functional.pad(x, (0, ALIGNMENT - C % ALIGNMENT))
    return ttnn.from_torch(x, device=device, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.ROW_MAJOR_LAYOUT)


def _run(device, c_in_block):
    """Build + run the patch-embed conv3d with an explicit c_in_block. Returns
    (torch_golden, tt_output_torch). Raises RuntimeError if the op overflows L1."""
    torch.manual_seed(42)
    N, C, D, H, W = INPUT_SHAPE
    D_out = _out_size(D, PADDING[0], STRIDE[0], KERNEL[0])
    H_out = _out_size(H, PADDING[1], STRIDE[1], KERNEL[1])
    W_out = _out_size(W, PADDING[2], STRIDE[2], KERNEL[2])

    x = torch.randn(N, C, D, H, W, dtype=torch.float32)
    conv = nn.Conv3d(C, OUT_CHANNELS, kernel_size=KERNEL, stride=STRIDE,
                     padding=PADDING, groups=GROUPS, bias=True)
    gt = conv(x)

    tt_input = _prepare_input(x, C, device)

    tt_weight = ttnn.from_torch(conv.weight.data, dtype=ttnn.DataType.BFLOAT16, pad_value=0)
    tt_weight = ttnn.experimental.prepare_conv3d_weights(
        weight_tensor=tt_weight, groups=GROUPS, C_in_block=c_in_block,
        alignment=ALIGNMENT, device=device,
    )
    tt_bias = ttnn.from_torch(
        conv.bias.data.reshape(1, -1), device=device,
        dtype=ttnn.DataType.BFLOAT16, layout=ttnn.TILE_LAYOUT, pad_value=0,
    )

    grid = device.compute_with_storage_grid_size()
    config = ttnn.Conv3dConfig(
        weights_dtype=ttnn.bfloat16,
        output_layout=ttnn.ROW_MAJOR_LAYOUT,
        T_out_block=1, W_out_block=1, H_out_block=1,
        C_out_block=32, C_in_block=c_in_block,
        dilation=(1, 1, 1),
        compute_with_storage_grid_size=grid,
    )
    kernel_config = ttnn.init_device_compute_kernel_config(
        device.arch(), math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False,
    )

    tt_out = ttnn.experimental.conv3d(
        input_tensor=tt_input, weight_tensor=tt_weight, device=device,
        bias_tensor=tt_bias, dtype=ttnn.bfloat16, output_channels=OUT_CHANNELS,
        kernel_size=KERNEL, stride=STRIDE, padding=PADDING, padding_mode="zeros",
        groups=GROUPS, config=config, compute_kernel_config=kernel_config,
    )
    tt_out = ttnn.to_torch(tt_out, device=device, dtype=torch.float32)
    tt_out = tt_out.reshape(N, D_out, H_out, W_out, OUT_CHANNELS).permute(0, 4, 1, 2, 3)
    return gt, tt_out


def _pcc(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    num = torch.sum((a - a.mean()) * (b - b.mean()))
    den = torch.sqrt(torch.sum((a - a.mean()) ** 2) * torch.sum((b - b.mean()) ** 2))
    return (num / den).item()


def main():
    device = ttnn.open_device(device_id=0)
    try:
        # c_in_block = 32 (what tt-mlir hard-codes) -> expected L1 overflow.
        print("=== c_in_block=32 (tt-mlir default) ===")
        try:
            _run(device, c_in_block=32)
            print("  UNEXPECTED: no overflow (expected CB > L1)")
        except RuntimeError as e:
            print(f"  THREW as expected: {str(e).splitlines()[0]}")

        # c_in_block = 16 (minimal valid block) -> expected to fit and match golden.
        print("=== c_in_block=16 (minimal valid) ===")
        gt, tt_out = _run(device, c_in_block=16)
        assert tt_out.shape == gt.shape, (tt_out.shape, gt.shape)
        pcc = _pcc(gt, tt_out)
        print(f"  PASSED (fits L1), PCC={pcc:.5f}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
