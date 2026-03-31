# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Z-Image Transformer: PyTorch vs TTNN comparison.

Runs the transformer on both PyTorch (CPU) and TTNN (TT device),
compares PCC to verify numerical accuracy.

Model: Tongyi-MAI/Z-Image transformer block
"""

import time

import torch
import ttnn
from model_pt import ZImageTransformerPT, get_input
from model_ttnn import ZImageTransformerTTNN
from utils import calculate_pcc


def main():
    # Load inputs (cached after first run)
    latent_input, timestep, cap_feat = get_input()

    # Load PyTorch model and get reference output
    model_pt = ZImageTransformerPT(latent_input, cap_feat)
    print("Running PyTorch reference...")
    with torch.inference_mode():
        output_pt = model_pt(latent_input, timestep, cap_feat)
    print(f"PyTorch output shape: {output_pt.shape}")

    # Convert inputs to TTNN host tensors
    latent_ttnn = ttnn.from_torch(latent_input)
    latent_ttnn = ttnn.to_dtype(latent_ttnn, ttnn.DataType.BFLOAT16)
    timestep_ttnn = ttnn.from_torch(timestep)
    timestep_ttnn = ttnn.to_dtype(timestep_ttnn, ttnn.DataType.FLOAT32)
    cap_feat_ttnn = ttnn.from_torch(cap_feat)
    cap_feat_ttnn = ttnn.to_dtype(cap_feat_ttnn, ttnn.DataType.BFLOAT16)

    # Open device
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape((1, 1)),
        l1_small_size=1 << 15,
    )

    # Load TTNN model from PyTorch weights
    model_ttnn = ZImageTransformerTTNN(mesh_device, model_pt.state_dict_for_ttnn())

    # Run TTNN model
    for i in range(3):
        start_time = time.time()

        out_device = model_ttnn.forward(latent_ttnn, timestep_ttnn, cap_feat_ttnn)
        out_host = ttnn.from_device(out_device, blocking=True)
        ttnn.synchronize_device(mesh_device)

        end_time = time.time()

        duration = (end_time - start_time) * 1000
        fps = 1.0 / (end_time - start_time)
        pcc = calculate_pcc(output_pt, ttnn.to_torch(out_host))

        print(f"Iteration {i}")
        print(f"\tDuration: {duration:.1f}ms")
        print(f"\tFPS: {fps:.2f}")
        print(f"\tPCC: {pcc:.6f}")


if __name__ == "__main__":
    main()
