# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# SELF-CONTAINED pure-TTNN repro of the fast-dispatch to_device wedge.
# See preamble in prior comments; wedges around "[graph_1 op 0238]".
#   $ python repro_fd_wedge.py                       # expect hang at op 0238
#   $ TT_METAL_SLOW_DISPATCH_MODE=1 python ...       # expect completion
import torch
import ttnn

SHAPES = [
    ((1, 1024), "INT32"),
    ((1,), "INT32"),
    ((64,), "FLOAT32"),
    ((1, 1), "INT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1, 1), "INT32"),
    ((151936, 1024), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((64,), "FLOAT32"),
    ((128,), "BFLOAT16"),
    ((4096, 1024), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1025, 8, 32, 128), "BFLOAT16"),
    ((1024, 2048), "BFLOAT16"),
    ((128,), "BFLOAT16"),
    ((1024, 3072), "BFLOAT16"),
    ((6144, 1024), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
    ((1024,), "BFLOAT16"),
]

DTYPE_MAP = {
    "BFLOAT16": (torch.bfloat16, ttnn.bfloat16),
    "FLOAT32": (torch.float32, ttnn.float32),
    "INT32": (torch.int32, ttnn.uint32),
}


def synthesize_inputs():
    inputs = []
    for shape, dt in SHAPES:
        torch_dt, ttnn_dt = DTYPE_MAP[dt]
        z = torch.zeros(shape, dtype=torch_dt)
        inputs.append(ttnn.from_torch(z, dtype=ttnn_dt, layout=ttnn.ROW_MAJOR_LAYOUT))
    return inputs


def open_device():
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape([1, 1]), l1_small_size=1 << 15
    )


def forward(input, device):
    print("[graph_1 op 0013] to_layout -> ttnn_to_layout_4", flush=True)
    ttnn_to_layout_4 = ttnn.to_layout(
        input[8], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0014] to_device -> ttnn_to_device_8", flush=True)
    ttnn_to_device_8 = ttnn.to_device(
        ttnn_to_layout_4,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0015] to_layout -> ttnn_to_layout_5", flush=True)
    ttnn_to_layout_5 = ttnn.to_layout(
        input[9], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0016] to_device -> ttnn_to_device_9", flush=True)
    ttnn_to_device_9 = ttnn.to_device(
        ttnn_to_layout_5,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0017] to_layout -> ttnn_to_layout_6", flush=True)
    ttnn_to_layout_6 = ttnn.to_layout(
        input[10], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0018] to_device -> ttnn_to_device_10", flush=True)
    ttnn_to_device_10 = ttnn.to_device(
        ttnn_to_layout_6,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0019] to_layout -> ttnn_to_layout_7", flush=True)
    ttnn_to_layout_7 = ttnn.to_layout(
        input[11], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0020] to_device -> ttnn_to_device_11", flush=True)
    ttnn_to_device_11 = ttnn.to_device(
        ttnn_to_layout_7,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0021] to_layout -> ttnn_to_layout_8", flush=True)
    ttnn_to_layout_8 = ttnn.to_layout(
        input[12], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0022] to_device -> ttnn_to_device_12", flush=True)
    ttnn_to_device_12 = ttnn.to_device(
        ttnn_to_layout_8,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0023] to_layout -> ttnn_to_layout_9", flush=True)
    ttnn_to_layout_9 = ttnn.to_layout(
        input[13], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0024] to_device -> ttnn_to_device_13", flush=True)
    ttnn_to_device_13 = ttnn.to_device(
        ttnn_to_layout_9,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0025] to_layout -> ttnn_to_layout_10", flush=True)
    ttnn_to_layout_10 = ttnn.to_layout(
        input[14], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0026] to_device -> ttnn_to_device_14", flush=True)
    ttnn_to_device_14 = ttnn.to_device(
        ttnn_to_layout_10,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0027] to_layout -> ttnn_to_layout_11", flush=True)
    ttnn_to_layout_11 = ttnn.to_layout(
        input[15], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0028] to_device -> ttnn_to_device_15", flush=True)
    ttnn_to_device_15 = ttnn.to_device(
        ttnn_to_layout_11,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0029] to_layout -> ttnn_to_layout_12", flush=True)
    ttnn_to_layout_12 = ttnn.to_layout(
        input[16], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0030] to_device -> ttnn_to_device_16", flush=True)
    ttnn_to_device_16 = ttnn.to_device(
        ttnn_to_layout_12,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0031] to_layout -> ttnn_to_layout_13", flush=True)
    ttnn_to_layout_13 = ttnn.to_layout(
        input[17], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0032] to_device -> ttnn_to_device_17", flush=True)
    ttnn_to_device_17 = ttnn.to_device(
        ttnn_to_layout_13,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0033] to_layout -> ttnn_to_layout_14", flush=True)
    ttnn_to_layout_14 = ttnn.to_layout(
        input[18], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0034] to_device -> ttnn_to_device_18", flush=True)
    ttnn_to_device_18 = ttnn.to_device(
        ttnn_to_layout_14,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0035] to_layout -> ttnn_to_layout_15", flush=True)
    ttnn_to_layout_15 = ttnn.to_layout(
        input[19], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0036] to_device -> ttnn_to_device_19", flush=True)
    ttnn_to_device_19 = ttnn.to_device(
        ttnn_to_layout_15,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0037] to_layout -> ttnn_to_layout_16", flush=True)
    ttnn_to_layout_16 = ttnn.to_layout(
        input[20], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0038] to_device -> ttnn_to_device_20", flush=True)
    ttnn_to_device_20 = ttnn.to_device(
        ttnn_to_layout_16,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0039] to_layout -> ttnn_to_layout_17", flush=True)
    ttnn_to_layout_17 = ttnn.to_layout(
        input[21], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0040] to_device -> ttnn_to_device_21", flush=True)
    ttnn_to_device_21 = ttnn.to_device(
        ttnn_to_layout_17,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0041] to_layout -> ttnn_to_layout_18", flush=True)
    ttnn_to_layout_18 = ttnn.to_layout(
        input[22], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0042] to_device -> ttnn_to_device_22", flush=True)
    ttnn_to_device_22 = ttnn.to_device(
        ttnn_to_layout_18,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0043] to_layout -> ttnn_to_layout_19", flush=True)
    ttnn_to_layout_19 = ttnn.to_layout(
        input[23], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0044] to_device -> ttnn_to_device_23", flush=True)
    ttnn_to_device_23 = ttnn.to_device(
        ttnn_to_layout_19,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0045] to_layout -> ttnn_to_layout_20", flush=True)
    ttnn_to_layout_20 = ttnn.to_layout(
        input[24], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0046] to_device -> ttnn_to_device_24", flush=True)
    ttnn_to_device_24 = ttnn.to_device(
        ttnn_to_layout_20,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0047] to_layout -> ttnn_to_layout_21", flush=True)
    ttnn_to_layout_21 = ttnn.to_layout(
        input[25], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0048] to_device -> ttnn_to_device_25", flush=True)
    ttnn_to_device_25 = ttnn.to_device(
        ttnn_to_layout_21,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0049] to_layout -> ttnn_to_layout_22", flush=True)
    ttnn_to_layout_22 = ttnn.to_layout(
        input[26], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0050] to_device -> ttnn_to_device_26", flush=True)
    ttnn_to_device_26 = ttnn.to_device(
        ttnn_to_layout_22,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0051] to_layout -> ttnn_to_layout_23", flush=True)
    ttnn_to_layout_23 = ttnn.to_layout(
        input[27], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0052] to_device -> ttnn_to_device_27", flush=True)
    ttnn_to_device_27 = ttnn.to_device(
        ttnn_to_layout_23,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0053] to_layout -> ttnn_to_layout_24", flush=True)
    ttnn_to_layout_24 = ttnn.to_layout(
        input[28], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0054] to_device -> ttnn_to_device_28", flush=True)
    ttnn_to_device_28 = ttnn.to_device(
        ttnn_to_layout_24,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0055] to_layout -> ttnn_to_layout_25", flush=True)
    ttnn_to_layout_25 = ttnn.to_layout(
        input[29], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0056] to_device -> ttnn_to_device_29", flush=True)
    ttnn_to_device_29 = ttnn.to_device(
        ttnn_to_layout_25,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0057] to_layout -> ttnn_to_layout_26", flush=True)
    ttnn_to_layout_26 = ttnn.to_layout(
        input[30], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0058] to_device -> ttnn_to_device_30", flush=True)
    ttnn_to_device_30 = ttnn.to_device(
        ttnn_to_layout_26,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0059] to_layout -> ttnn_to_layout_27", flush=True)
    ttnn_to_layout_27 = ttnn.to_layout(
        input[31], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0060] to_device -> ttnn_to_device_31", flush=True)
    ttnn_to_device_31 = ttnn.to_device(
        ttnn_to_layout_27,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0061] to_layout -> ttnn_to_layout_28", flush=True)
    ttnn_to_layout_28 = ttnn.to_layout(
        input[32], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0062] to_device -> ttnn_to_device_32", flush=True)
    ttnn_to_device_32 = ttnn.to_device(
        ttnn_to_layout_28,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0063] to_layout -> ttnn_to_layout_29", flush=True)
    ttnn_to_layout_29 = ttnn.to_layout(
        input[33], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0064] to_device -> ttnn_to_device_33", flush=True)
    ttnn_to_device_33 = ttnn.to_device(
        ttnn_to_layout_29,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0065] to_layout -> ttnn_to_layout_30", flush=True)
    ttnn_to_layout_30 = ttnn.to_layout(
        input[34], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0066] to_device -> ttnn_to_device_34", flush=True)
    ttnn_to_device_34 = ttnn.to_device(
        ttnn_to_layout_30,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0067] to_layout -> ttnn_to_layout_31", flush=True)
    ttnn_to_layout_31 = ttnn.to_layout(
        input[35], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0068] to_device -> ttnn_to_device_35", flush=True)
    ttnn_to_device_35 = ttnn.to_device(
        ttnn_to_layout_31,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0069] to_layout -> ttnn_to_layout_32", flush=True)
    ttnn_to_layout_32 = ttnn.to_layout(
        input[36], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0070] to_device -> ttnn_to_device_36", flush=True)
    ttnn_to_device_36 = ttnn.to_device(
        ttnn_to_layout_32,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0071] to_layout -> ttnn_to_layout_33", flush=True)
    ttnn_to_layout_33 = ttnn.to_layout(
        input[37], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0072] to_device -> ttnn_to_device_37", flush=True)
    ttnn_to_device_37 = ttnn.to_device(
        ttnn_to_layout_33,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0073] to_layout -> ttnn_to_layout_34", flush=True)
    ttnn_to_layout_34 = ttnn.to_layout(
        input[38], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0074] to_device -> ttnn_to_device_38", flush=True)
    ttnn_to_device_38 = ttnn.to_device(
        ttnn_to_layout_34,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0075] to_layout -> ttnn_to_layout_35", flush=True)
    ttnn_to_layout_35 = ttnn.to_layout(
        input[39], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0076] to_device -> ttnn_to_device_39", flush=True)
    ttnn_to_device_39 = ttnn.to_device(
        ttnn_to_layout_35,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0077] to_layout -> ttnn_to_layout_36", flush=True)
    ttnn_to_layout_36 = ttnn.to_layout(
        input[40], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0078] to_device -> ttnn_to_device_40", flush=True)
    ttnn_to_device_40 = ttnn.to_device(
        ttnn_to_layout_36,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0079] to_layout -> ttnn_to_layout_37", flush=True)
    ttnn_to_layout_37 = ttnn.to_layout(
        input[41], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0080] to_device -> ttnn_to_device_41", flush=True)
    ttnn_to_device_41 = ttnn.to_device(
        ttnn_to_layout_37,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0081] to_layout -> ttnn_to_layout_38", flush=True)
    ttnn_to_layout_38 = ttnn.to_layout(
        input[42], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0082] to_device -> ttnn_to_device_42", flush=True)
    ttnn_to_device_42 = ttnn.to_device(
        ttnn_to_layout_38,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0083] to_layout -> ttnn_to_layout_39", flush=True)
    ttnn_to_layout_39 = ttnn.to_layout(
        input[43], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0084] to_device -> ttnn_to_device_43", flush=True)
    ttnn_to_device_43 = ttnn.to_device(
        ttnn_to_layout_39,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0085] to_layout -> ttnn_to_layout_40", flush=True)
    ttnn_to_layout_40 = ttnn.to_layout(
        input[44], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0086] to_device -> ttnn_to_device_44", flush=True)
    ttnn_to_device_44 = ttnn.to_device(
        ttnn_to_layout_40,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0087] to_layout -> ttnn_to_layout_41", flush=True)
    ttnn_to_layout_41 = ttnn.to_layout(
        input[45], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0088] to_device -> ttnn_to_device_45", flush=True)
    ttnn_to_device_45 = ttnn.to_device(
        ttnn_to_layout_41,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0089] to_layout -> ttnn_to_layout_42", flush=True)
    ttnn_to_layout_42 = ttnn.to_layout(
        input[46], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0090] to_device -> ttnn_to_device_46", flush=True)
    ttnn_to_device_46 = ttnn.to_device(
        ttnn_to_layout_42,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0091] to_layout -> ttnn_to_layout_43", flush=True)
    ttnn_to_layout_43 = ttnn.to_layout(
        input[47], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0092] to_device -> ttnn_to_device_47", flush=True)
    ttnn_to_device_47 = ttnn.to_device(
        ttnn_to_layout_43,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0093] to_layout -> ttnn_to_layout_44", flush=True)
    ttnn_to_layout_44 = ttnn.to_layout(
        input[48], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0094] to_device -> ttnn_to_device_48", flush=True)
    ttnn_to_device_48 = ttnn.to_device(
        ttnn_to_layout_44,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0095] to_layout -> ttnn_to_layout_45", flush=True)
    ttnn_to_layout_45 = ttnn.to_layout(
        input[49], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0096] to_device -> ttnn_to_device_49", flush=True)
    ttnn_to_device_49 = ttnn.to_device(
        ttnn_to_layout_45,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0097] to_layout -> ttnn_to_layout_46", flush=True)
    ttnn_to_layout_46 = ttnn.to_layout(
        input[50], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0098] to_device -> ttnn_to_device_50", flush=True)
    ttnn_to_device_50 = ttnn.to_device(
        ttnn_to_layout_46,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0099] to_layout -> ttnn_to_layout_47", flush=True)
    ttnn_to_layout_47 = ttnn.to_layout(
        input[51], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0100] to_device -> ttnn_to_device_51", flush=True)
    ttnn_to_device_51 = ttnn.to_device(
        ttnn_to_layout_47,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0101] to_layout -> ttnn_to_layout_48", flush=True)
    ttnn_to_layout_48 = ttnn.to_layout(
        input[52], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0102] to_device -> ttnn_to_device_52", flush=True)
    ttnn_to_device_52 = ttnn.to_device(
        ttnn_to_layout_48,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0103] to_layout -> ttnn_to_layout_49", flush=True)
    ttnn_to_layout_49 = ttnn.to_layout(
        input[53], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0104] to_device -> ttnn_to_device_53", flush=True)
    ttnn_to_device_53 = ttnn.to_device(
        ttnn_to_layout_49,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0105] to_layout -> ttnn_to_layout_50", flush=True)
    ttnn_to_layout_50 = ttnn.to_layout(
        input[54], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0106] to_device -> ttnn_to_device_54", flush=True)
    ttnn_to_device_54 = ttnn.to_device(
        ttnn_to_layout_50,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0107] to_layout -> ttnn_to_layout_51", flush=True)
    ttnn_to_layout_51 = ttnn.to_layout(
        input[55], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0108] to_device -> ttnn_to_device_55", flush=True)
    ttnn_to_device_55 = ttnn.to_device(
        ttnn_to_layout_51,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0109] to_layout -> ttnn_to_layout_52", flush=True)
    ttnn_to_layout_52 = ttnn.to_layout(
        input[56], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0110] to_device -> ttnn_to_device_56", flush=True)
    ttnn_to_device_56 = ttnn.to_device(
        ttnn_to_layout_52,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0111] to_layout -> ttnn_to_layout_53", flush=True)
    ttnn_to_layout_53 = ttnn.to_layout(
        input[57], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0112] to_device -> ttnn_to_device_57", flush=True)
    ttnn_to_device_57 = ttnn.to_device(
        ttnn_to_layout_53,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0113] to_layout -> ttnn_to_layout_54", flush=True)
    ttnn_to_layout_54 = ttnn.to_layout(
        input[58], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0114] to_device -> ttnn_to_device_58", flush=True)
    ttnn_to_device_58 = ttnn.to_device(
        ttnn_to_layout_54,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0115] to_layout -> ttnn_to_layout_55", flush=True)
    ttnn_to_layout_55 = ttnn.to_layout(
        input[59], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0116] to_device -> ttnn_to_device_59", flush=True)
    ttnn_to_device_59 = ttnn.to_device(
        ttnn_to_layout_55,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0117] to_layout -> ttnn_to_layout_56", flush=True)
    ttnn_to_layout_56 = ttnn.to_layout(
        input[60], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0118] to_device -> ttnn_to_device_60", flush=True)
    ttnn_to_device_60 = ttnn.to_device(
        ttnn_to_layout_56,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0119] to_layout -> ttnn_to_layout_57", flush=True)
    ttnn_to_layout_57 = ttnn.to_layout(
        input[61], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0120] to_device -> ttnn_to_device_61", flush=True)
    ttnn_to_device_61 = ttnn.to_device(
        ttnn_to_layout_57,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0121] to_layout -> ttnn_to_layout_58", flush=True)
    ttnn_to_layout_58 = ttnn.to_layout(
        input[62], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0122] to_device -> ttnn_to_device_62", flush=True)
    ttnn_to_device_62 = ttnn.to_device(
        ttnn_to_layout_58,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0123] to_layout -> ttnn_to_layout_59", flush=True)
    ttnn_to_layout_59 = ttnn.to_layout(
        input[63], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0124] to_device -> ttnn_to_device_63", flush=True)
    ttnn_to_device_63 = ttnn.to_device(
        ttnn_to_layout_59,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0125] to_layout -> ttnn_to_layout_60", flush=True)
    ttnn_to_layout_60 = ttnn.to_layout(
        input[64], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0126] to_device -> ttnn_to_device_64", flush=True)
    ttnn_to_device_64 = ttnn.to_device(
        ttnn_to_layout_60,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0127] to_layout -> ttnn_to_layout_61", flush=True)
    ttnn_to_layout_61 = ttnn.to_layout(
        input[65], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0128] to_device -> ttnn_to_device_65", flush=True)
    ttnn_to_device_65 = ttnn.to_device(
        ttnn_to_layout_61,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0129] to_layout -> ttnn_to_layout_62", flush=True)
    ttnn_to_layout_62 = ttnn.to_layout(
        input[66], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0130] to_device -> ttnn_to_device_66", flush=True)
    ttnn_to_device_66 = ttnn.to_device(
        ttnn_to_layout_62,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0131] to_layout -> ttnn_to_layout_63", flush=True)
    ttnn_to_layout_63 = ttnn.to_layout(
        input[67], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0132] to_device -> ttnn_to_device_67", flush=True)
    ttnn_to_device_67 = ttnn.to_device(
        ttnn_to_layout_63,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0133] to_layout -> ttnn_to_layout_64", flush=True)
    ttnn_to_layout_64 = ttnn.to_layout(
        input[68], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0134] to_device -> ttnn_to_device_68", flush=True)
    ttnn_to_device_68 = ttnn.to_device(
        ttnn_to_layout_64,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0135] to_layout -> ttnn_to_layout_65", flush=True)
    ttnn_to_layout_65 = ttnn.to_layout(
        input[69], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0136] to_device -> ttnn_to_device_69", flush=True)
    ttnn_to_device_69 = ttnn.to_device(
        ttnn_to_layout_65,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0137] to_layout -> ttnn_to_layout_66", flush=True)
    ttnn_to_layout_66 = ttnn.to_layout(
        input[70], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0138] to_device -> ttnn_to_device_70", flush=True)
    ttnn_to_device_70 = ttnn.to_device(
        ttnn_to_layout_66,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0139] to_layout -> ttnn_to_layout_67", flush=True)
    ttnn_to_layout_67 = ttnn.to_layout(
        input[71], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0140] to_device -> ttnn_to_device_71", flush=True)
    ttnn_to_device_71 = ttnn.to_device(
        ttnn_to_layout_67,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0141] to_layout -> ttnn_to_layout_68", flush=True)
    ttnn_to_layout_68 = ttnn.to_layout(
        input[72], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0142] to_device -> ttnn_to_device_72", flush=True)
    ttnn_to_device_72 = ttnn.to_device(
        ttnn_to_layout_68,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0143] to_layout -> ttnn_to_layout_69", flush=True)
    ttnn_to_layout_69 = ttnn.to_layout(
        input[73], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0144] to_device -> ttnn_to_device_73", flush=True)
    ttnn_to_device_73 = ttnn.to_device(
        ttnn_to_layout_69,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0145] to_layout -> ttnn_to_layout_70", flush=True)
    ttnn_to_layout_70 = ttnn.to_layout(
        input[74], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0146] to_device -> ttnn_to_device_74", flush=True)
    ttnn_to_device_74 = ttnn.to_device(
        ttnn_to_layout_70,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0147] to_layout -> ttnn_to_layout_71", flush=True)
    ttnn_to_layout_71 = ttnn.to_layout(
        input[75], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0148] to_device -> ttnn_to_device_75", flush=True)
    ttnn_to_device_75 = ttnn.to_device(
        ttnn_to_layout_71,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0149] to_layout -> ttnn_to_layout_72", flush=True)
    ttnn_to_layout_72 = ttnn.to_layout(
        input[76], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0150] to_device -> ttnn_to_device_76", flush=True)
    ttnn_to_device_76 = ttnn.to_device(
        ttnn_to_layout_72,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0151] to_layout -> ttnn_to_layout_73", flush=True)
    ttnn_to_layout_73 = ttnn.to_layout(
        input[77], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0152] to_device -> ttnn_to_device_77", flush=True)
    ttnn_to_device_77 = ttnn.to_device(
        ttnn_to_layout_73,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0153] to_layout -> ttnn_to_layout_74", flush=True)
    ttnn_to_layout_74 = ttnn.to_layout(
        input[78], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0154] to_device -> ttnn_to_device_78", flush=True)
    ttnn_to_device_78 = ttnn.to_device(
        ttnn_to_layout_74,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0155] to_layout -> ttnn_to_layout_75", flush=True)
    ttnn_to_layout_75 = ttnn.to_layout(
        input[79], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0156] to_device -> ttnn_to_device_79", flush=True)
    ttnn_to_device_79 = ttnn.to_device(
        ttnn_to_layout_75,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0157] to_layout -> ttnn_to_layout_76", flush=True)
    ttnn_to_layout_76 = ttnn.to_layout(
        input[80], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0158] to_device -> ttnn_to_device_80", flush=True)
    ttnn_to_device_80 = ttnn.to_device(
        ttnn_to_layout_76,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0159] to_layout -> ttnn_to_layout_77", flush=True)
    ttnn_to_layout_77 = ttnn.to_layout(
        input[81], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0160] to_device -> ttnn_to_device_81", flush=True)
    ttnn_to_device_81 = ttnn.to_device(
        ttnn_to_layout_77,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0161] to_layout -> ttnn_to_layout_78", flush=True)
    ttnn_to_layout_78 = ttnn.to_layout(
        input[82], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0162] to_device -> ttnn_to_device_82", flush=True)
    ttnn_to_device_82 = ttnn.to_device(
        ttnn_to_layout_78,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0163] to_layout -> ttnn_to_layout_79", flush=True)
    ttnn_to_layout_79 = ttnn.to_layout(
        input[83], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0164] to_device -> ttnn_to_device_83", flush=True)
    ttnn_to_device_83 = ttnn.to_device(
        ttnn_to_layout_79,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0165] to_layout -> ttnn_to_layout_80", flush=True)
    ttnn_to_layout_80 = ttnn.to_layout(
        input[84], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0166] to_device -> ttnn_to_device_84", flush=True)
    ttnn_to_device_84 = ttnn.to_device(
        ttnn_to_layout_80,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0167] to_layout -> ttnn_to_layout_81", flush=True)
    ttnn_to_layout_81 = ttnn.to_layout(
        input[85], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0168] to_device -> ttnn_to_device_85", flush=True)
    ttnn_to_device_85 = ttnn.to_device(
        ttnn_to_layout_81,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0169] to_layout -> ttnn_to_layout_82", flush=True)
    ttnn_to_layout_82 = ttnn.to_layout(
        input[86], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0170] to_device -> ttnn_to_device_86", flush=True)
    ttnn_to_device_86 = ttnn.to_device(
        ttnn_to_layout_82,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0171] to_layout -> ttnn_to_layout_83", flush=True)
    ttnn_to_layout_83 = ttnn.to_layout(
        input[87], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0172] to_device -> ttnn_to_device_87", flush=True)
    ttnn_to_device_87 = ttnn.to_device(
        ttnn_to_layout_83,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0173] to_layout -> ttnn_to_layout_84", flush=True)
    ttnn_to_layout_84 = ttnn.to_layout(
        input[88], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0174] to_device -> ttnn_to_device_88", flush=True)
    ttnn_to_device_88 = ttnn.to_device(
        ttnn_to_layout_84,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0175] to_layout -> ttnn_to_layout_85", flush=True)
    ttnn_to_layout_85 = ttnn.to_layout(
        input[89], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0176] to_device -> ttnn_to_device_89", flush=True)
    ttnn_to_device_89 = ttnn.to_device(
        ttnn_to_layout_85,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0177] to_layout -> ttnn_to_layout_86", flush=True)
    ttnn_to_layout_86 = ttnn.to_layout(
        input[90], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0178] to_device -> ttnn_to_device_90", flush=True)
    ttnn_to_device_90 = ttnn.to_device(
        ttnn_to_layout_86,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0179] to_layout -> ttnn_to_layout_87", flush=True)
    ttnn_to_layout_87 = ttnn.to_layout(
        input[91], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0180] to_device -> ttnn_to_device_91", flush=True)
    ttnn_to_device_91 = ttnn.to_device(
        ttnn_to_layout_87,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0181] to_layout -> ttnn_to_layout_88", flush=True)
    ttnn_to_layout_88 = ttnn.to_layout(
        input[92], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0182] to_device -> ttnn_to_device_92", flush=True)
    ttnn_to_device_92 = ttnn.to_device(
        ttnn_to_layout_88,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0183] to_layout -> ttnn_to_layout_89", flush=True)
    ttnn_to_layout_89 = ttnn.to_layout(
        input[93], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0184] to_device -> ttnn_to_device_93", flush=True)
    ttnn_to_device_93 = ttnn.to_device(
        ttnn_to_layout_89,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0185] to_layout -> ttnn_to_layout_90", flush=True)
    ttnn_to_layout_90 = ttnn.to_layout(
        input[94], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0186] to_device -> ttnn_to_device_94", flush=True)
    ttnn_to_device_94 = ttnn.to_device(
        ttnn_to_layout_90,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0187] to_layout -> ttnn_to_layout_91", flush=True)
    ttnn_to_layout_91 = ttnn.to_layout(
        input[95], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0188] to_device -> ttnn_to_device_95", flush=True)
    ttnn_to_device_95 = ttnn.to_device(
        ttnn_to_layout_91,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0189] to_layout -> ttnn_to_layout_92", flush=True)
    ttnn_to_layout_92 = ttnn.to_layout(
        input[96], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0190] to_device -> ttnn_to_device_96", flush=True)
    ttnn_to_device_96 = ttnn.to_device(
        ttnn_to_layout_92,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0191] to_layout -> ttnn_to_layout_93", flush=True)
    ttnn_to_layout_93 = ttnn.to_layout(
        input[97], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0192] to_device -> ttnn_to_device_97", flush=True)
    ttnn_to_device_97 = ttnn.to_device(
        ttnn_to_layout_93,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0193] to_layout -> ttnn_to_layout_94", flush=True)
    ttnn_to_layout_94 = ttnn.to_layout(
        input[98], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0194] to_device -> ttnn_to_device_98", flush=True)
    ttnn_to_device_98 = ttnn.to_device(
        ttnn_to_layout_94,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0195] to_layout -> ttnn_to_layout_95", flush=True)
    ttnn_to_layout_95 = ttnn.to_layout(
        input[99], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0196] to_device -> ttnn_to_device_99", flush=True)
    ttnn_to_device_99 = ttnn.to_device(
        ttnn_to_layout_95,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0197] to_layout -> ttnn_to_layout_96", flush=True)
    ttnn_to_layout_96 = ttnn.to_layout(
        input[100], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0198] to_device -> ttnn_to_device_100", flush=True)
    ttnn_to_device_100 = ttnn.to_device(
        ttnn_to_layout_96,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0199] to_layout -> ttnn_to_layout_97", flush=True)
    ttnn_to_layout_97 = ttnn.to_layout(
        input[101], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0200] to_device -> ttnn_to_device_101", flush=True)
    ttnn_to_device_101 = ttnn.to_device(
        ttnn_to_layout_97,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0201] to_layout -> ttnn_to_layout_98", flush=True)
    ttnn_to_layout_98 = ttnn.to_layout(
        input[102], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0202] to_device -> ttnn_to_device_102", flush=True)
    ttnn_to_device_102 = ttnn.to_device(
        ttnn_to_layout_98,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0203] to_layout -> ttnn_to_layout_99", flush=True)
    ttnn_to_layout_99 = ttnn.to_layout(
        input[103], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0204] to_device -> ttnn_to_device_103", flush=True)
    ttnn_to_device_103 = ttnn.to_device(
        ttnn_to_layout_99,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0205] to_layout -> ttnn_to_layout_100", flush=True)
    ttnn_to_layout_100 = ttnn.to_layout(
        input[104], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0206] to_device -> ttnn_to_device_104", flush=True)
    ttnn_to_device_104 = ttnn.to_device(
        ttnn_to_layout_100,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0207] to_layout -> ttnn_to_layout_101", flush=True)
    ttnn_to_layout_101 = ttnn.to_layout(
        input[105], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0208] to_device -> ttnn_to_device_105", flush=True)
    ttnn_to_device_105 = ttnn.to_device(
        ttnn_to_layout_101,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0209] to_layout -> ttnn_to_layout_102", flush=True)
    ttnn_to_layout_102 = ttnn.to_layout(
        input[106], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0210] to_device -> ttnn_to_device_106", flush=True)
    ttnn_to_device_106 = ttnn.to_device(
        ttnn_to_layout_102,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0211] to_layout -> ttnn_to_layout_103", flush=True)
    ttnn_to_layout_103 = ttnn.to_layout(
        input[107], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0212] to_device -> ttnn_to_device_107", flush=True)
    ttnn_to_device_107 = ttnn.to_device(
        ttnn_to_layout_103,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0213] to_layout -> ttnn_to_layout_104", flush=True)
    ttnn_to_layout_104 = ttnn.to_layout(
        input[108], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0214] to_device -> ttnn_to_device_108", flush=True)
    ttnn_to_device_108 = ttnn.to_device(
        ttnn_to_layout_104,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0215] to_layout -> ttnn_to_layout_105", flush=True)
    ttnn_to_layout_105 = ttnn.to_layout(
        input[109], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0216] to_device -> ttnn_to_device_109", flush=True)
    ttnn_to_device_109 = ttnn.to_device(
        ttnn_to_layout_105,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0217] to_layout -> ttnn_to_layout_106", flush=True)
    ttnn_to_layout_106 = ttnn.to_layout(
        input[110], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0218] to_device -> ttnn_to_device_110", flush=True)
    ttnn_to_device_110 = ttnn.to_device(
        ttnn_to_layout_106,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0219] to_layout -> ttnn_to_layout_107", flush=True)
    ttnn_to_layout_107 = ttnn.to_layout(
        input[111], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0220] to_device -> ttnn_to_device_111", flush=True)
    ttnn_to_device_111 = ttnn.to_device(
        ttnn_to_layout_107,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0221] to_layout -> ttnn_to_layout_108", flush=True)
    ttnn_to_layout_108 = ttnn.to_layout(
        input[112], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0222] to_device -> ttnn_to_device_112", flush=True)
    ttnn_to_device_112 = ttnn.to_device(
        ttnn_to_layout_108,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0223] to_layout -> ttnn_to_layout_109", flush=True)
    ttnn_to_layout_109 = ttnn.to_layout(
        input[113], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0224] to_device -> ttnn_to_device_113", flush=True)
    ttnn_to_device_113 = ttnn.to_device(
        ttnn_to_layout_109,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0225] to_layout -> ttnn_to_layout_110", flush=True)
    ttnn_to_layout_110 = ttnn.to_layout(
        input[114], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0226] to_device -> ttnn_to_device_114", flush=True)
    ttnn_to_device_114 = ttnn.to_device(
        ttnn_to_layout_110,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0227] to_layout -> ttnn_to_layout_111", flush=True)
    ttnn_to_layout_111 = ttnn.to_layout(
        input[115], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0228] to_device -> ttnn_to_device_115", flush=True)
    ttnn_to_device_115 = ttnn.to_device(
        ttnn_to_layout_111,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0229] to_layout -> ttnn_to_layout_112", flush=True)
    ttnn_to_layout_112 = ttnn.to_layout(
        input[116], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0230] to_device -> ttnn_to_device_116", flush=True)
    ttnn_to_device_116 = ttnn.to_device(
        ttnn_to_layout_112,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0231] to_layout -> ttnn_to_layout_113", flush=True)
    ttnn_to_layout_113 = ttnn.to_layout(
        input[117], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0232] to_device -> ttnn_to_device_117", flush=True)
    ttnn_to_device_117 = ttnn.to_device(
        ttnn_to_layout_113,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0233] to_layout -> ttnn_to_layout_114", flush=True)
    ttnn_to_layout_114 = ttnn.to_layout(
        input[118], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0234] to_device -> ttnn_to_device_118", flush=True)
    ttnn_to_device_118 = ttnn.to_device(
        ttnn_to_layout_114,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0235] to_layout -> ttnn_to_layout_115", flush=True)
    ttnn_to_layout_115 = ttnn.to_layout(
        input[119], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0236] to_device -> ttnn_to_device_119", flush=True)
    ttnn_to_device_119 = ttnn.to_device(
        ttnn_to_layout_115,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0237] to_layout -> ttnn_to_layout_116", flush=True)
    ttnn_to_layout_116 = ttnn.to_layout(
        input[120], ttnn.Layout.TILE, None, memory_config=None
    )
    ttnn.synchronize_device(device)
    print("[graph_1 op 0238] to_device -> ttnn_to_device_120", flush=True)
    ttnn_to_device_120 = ttnn.to_device(
        ttnn_to_layout_116,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
        ),
    )
    ttnn.synchronize_device(device)
    return None


if __name__ == "__main__":
    device = open_device()
    print("device open; synthesizing 314 zero-input tensors", flush=True)
    inputs = synthesize_inputs()
    print("inputs synthesized; entering chopped forward()", flush=True)
    forward(inputs, device)
    ttnn.synchronize_device(device)
    print("FORWARD DONE (no hang)", flush=True)
    ttnn.close_mesh_device(device)
