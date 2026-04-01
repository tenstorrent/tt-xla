# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the refactored TTNN VAE decoder.

Self-contained -- no imports from sibling directories.

Steps:
    1. Load PyTorch reference model (VaeDecoderPT)
    2. Get sample input and run PyTorch reference
    3. Open TTNN device
    4. Load weights from PyTorch state dict into TTNN tensors
    5. Build TTNN model (runs const-evals internally)
    6. Prepare input for TTNN
    7. Run TTNN model
    8. Compare outputs via PCC
    9. Print results
"""

import torch
import ttnn

from model_pt import VaeDecoderPT, get_input, SCALING_FACTOR, SHIFT_FACTOR
from model_ttnn import build_model, load_weights_from_pytorch, DRAM
from utils import calculate_pcc


def main():
    # ------------------------------------------------------------------
    # 1. Load PyTorch reference model
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Step 1: Loading PyTorch reference model...")
    print("=" * 60)
    pt_model = VaeDecoderPT()
    pt_state_dict = pt_model.state_dict_for_ttnn()
    print(f"  Loaded {len(pt_state_dict)} weight tensors")

    # ------------------------------------------------------------------
    # 2. Get sample input and run PyTorch reference
    # ------------------------------------------------------------------
    print("\nStep 2: Running PyTorch reference...")
    latent = get_input()  # [1, 16, 160, 90] bfloat16
    print(f"  Input shape: {latent.shape}, dtype: {latent.dtype}")

    with torch.no_grad():
        # Apply VAE scaling (same as VaeDecoderPT.forward)
        scaled_latent = latent / SCALING_FACTOR + SHIFT_FACTOR
        pt_output = pt_model.decoder(scaled_latent)
    print(f"  PyTorch output shape: {pt_output.shape}")

    # ------------------------------------------------------------------
    # 3. Open TTNN device
    # ------------------------------------------------------------------
    print("\nStep 3: Opening TTNN device...")
    device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape([1, 1]),
        l1_small_size=1 << 15,
    )
    print(f"  Device: {device}")

    try:
        # ------------------------------------------------------------------
        # 4. Load weights into TTNN tensors
        # ------------------------------------------------------------------
        print("\nStep 4: Loading weights into TTNN...")
        weights = load_weights_from_pytorch(pt_state_dict, device)
        print(f"  Loaded {len(weights)} TTNN weight tensors")

        # ------------------------------------------------------------------
        # 5. Build TTNN model (runs const-evals)
        # ------------------------------------------------------------------
        print("\nStep 5: Building TTNN model (const-evals)...")
        model = build_model(weights, device)

        # ------------------------------------------------------------------
        # 6. Prepare input for TTNN
        # ------------------------------------------------------------------
        print("\nStep 6: Preparing input tensor...")
        # Apply VAE scaling on CPU
        ttnn_input_pt = scaled_latent.contiguous()
        ttnn_input = ttnn.from_torch(
            ttnn_input_pt, dtype=ttnn.DataType.BFLOAT16, layout=ttnn.Layout.TILE
        )
        ttnn_input = ttnn.to_device(ttnn_input, device, memory_config=DRAM)
        print(f"  TTNN input shape: {ttnn_input.shape}")

        # ------------------------------------------------------------------
        # 7. Run TTNN model
        # ------------------------------------------------------------------
        print("\nStep 7: Running TTNN model...")
        ttnn_output = model.forward(ttnn_input)
        print(f"  TTNN output shape: {ttnn_output.shape}")

        # ------------------------------------------------------------------
        # 8. Compare outputs
        # ------------------------------------------------------------------
        print("\nStep 8: Comparing outputs...")
        ttnn_output_pt = ttnn.to_torch(ttnn_output).float()
        pt_output_float = pt_output.float()

        pcc = calculate_pcc(pt_output_float, ttnn_output_pt)
        print(f"  PCC: {pcc.item():.6f}")

        # ------------------------------------------------------------------
        # 9. Print results
        # ------------------------------------------------------------------
        print("\n" + "=" * 60)
        print("Results:")
        print(f"  PyTorch output:  shape={pt_output.shape}")
        print(f"  TTNN output:     shape={ttnn_output_pt.shape}")
        print(f"  PCC:             {pcc.item():.6f}")
        if pcc.item() > 0.99:
            print("  Status:          PASS (PCC > 0.99)")
        elif pcc.item() > 0.95:
            print("  Status:          MARGINAL (0.95 < PCC < 0.99)")
        else:
            print("  Status:          FAIL (PCC < 0.95)")
        print("=" * 60)

    finally:
        # ------------------------------------------------------------------
        # Cleanup
        # ------------------------------------------------------------------
        print("\nClosing device...")
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
