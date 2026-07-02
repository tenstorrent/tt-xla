# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# Order-faithful + preload-all variant: matches _repro_forward exactly but
# without the generated forward() function -- preload all host tensors, keep
# them alive, then replay graph_1's exact upload order, async, resident.
import json
import os

import ttnn

BASE = os.path.dirname(os.path.abspath(__file__))
TENSORS = os.path.join(BASE, "qwen_codegen", "graph_1", "tensors")
ORDER = json.load(
    open(os.path.join(BASE, "qwen_codegen", "graph_1", "upload_order.json"))
)
N = len(
    [f for f in os.listdir(TENSORS) if f.startswith("arg") and f.endswith(".tensorbin")]
)
DRAM = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None
)
ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)
device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape([1, 1]), l1_small_size=1 << 15)
device.enable_program_cache()
print(f"opened {device}", flush=True)
# preload ALL host tensors upfront, keep alive (like _repro_forward's `inp`)
inp = [ttnn.load_tensor(f"{TENSORS}/arg{k}.tensorbin") for k in range(N)]
print(f"preloaded {N} host tensors", flush=True)
resident = []
for i, (k, tilize) in enumerate(ORDER):
    t = inp[k]
    if tilize and t.layout != ttnn.Layout.TILE:
        t = ttnn.to_layout(t, ttnn.Layout.TILE)
    resident.append(ttnn.to_device(t, device, memory_config=DRAM))
    print(f"[{i:04d}] input[{k}] -> to_device ok", flush=True)
ttnn.synchronize_device(device)
print("ALL UPLOADS DONE (no hang)", flush=True)
ttnn.close_mesh_device(device)
