# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Z-Image Text Encoder: PyTorch vs TTNN comparison.

Runs the Qwen3 text encoder on both PyTorch (CPU) and TTNN (TT device),
compares PCC to verify numerical accuracy.

Model: Tongyi-MAI/Z-Image text encoder (Qwen3-based, 35 decoder layers)
"""

import time

import torch
import ttnn
from model_pt import TextEncoderPT, get_input
from model_ttnn import TextEncoderTTNN
from utils import calculate_pcc


def main():
    # Load inputs
    input_ids, attention_mask = get_input()
    print(f"Input IDs shape: {input_ids.shape}, Attention mask shape: {attention_mask.shape}")

    # Load PyTorch model and get reference output
    model_pt = TextEncoderPT()
    print("Running PyTorch reference...")
    output_pt = model_pt(input_ids, attention_mask)
    print(f"PyTorch output shape: {output_pt.shape}")

    # Convert inputs to TTNN host tensors
    input_ids_ttnn = ttnn.from_torch(input_ids.int())  # INT32
    input_ids_ttnn = ttnn.to_dtype(input_ids_ttnn, ttnn.DataType.INT32)
    attention_mask_ttnn = ttnn.from_torch(attention_mask)
    attention_mask_ttnn = ttnn.to_dtype(attention_mask_ttnn, ttnn.DataType.BFLOAT16)

    # Open device
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape((1, 1)),
        l1_small_size=1 << 15,
    )

    # Move inputs to device
    dram = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
    )
    input_ids_dev = ttnn.to_device(input_ids_ttnn, mesh_device, memory_config=dram)
    attention_mask_dev = ttnn.to_device(attention_mask_ttnn, mesh_device, memory_config=dram)

    # Load TTNN model from PyTorch weights
    print("Building TTNN model...")
    model_ttnn = TextEncoderTTNN(mesh_device, model_pt.state_dict_for_ttnn())

    # Compute non-padding token mask for focused PCC
    non_pad_mask = attention_mask[0] > 0  # [512] bool
    num_real_tokens = int(non_pad_mask.sum().item())

    # Run TTNN model
    for i in range(3):
        start_time = time.time()

        out_device = model_ttnn.forward(input_ids_dev, attention_mask_dev)
        out_host = ttnn.from_device(out_device, blocking=True)
        ttnn.synchronize_device(mesh_device)

        end_time = time.time()

        duration = (end_time - start_time) * 1000
        fps = 1.0 / (end_time - start_time)
        output_ttnn = ttnn.to_torch(out_host)
        pcc = calculate_pcc(output_pt, output_ttnn)

        # PCC on non-padding tokens only (the meaningful content)
        pcc_content = calculate_pcc(
            output_pt[0, non_pad_mask, :],
            output_ttnn[0, non_pad_mask, :],
        )

        print(f"Iteration {i}")
        print(f"\tDuration: {duration:.1f}ms")
        print(f"\tFPS: {fps:.2f}")
        print(f"\tPCC (all tokens): {pcc:.6f}")
        print(f"\tPCC (content tokens, n={num_real_tokens}): {pcc_content:.6f}")
        print(f"\tOutput shape: {output_ttnn.shape}")


if __name__ == "__main__":
    main()
