# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Check what sharding spec tensors have after different operations."""

import os

os.environ["XLA_STABLEHLO_COMPILE"] = "1"

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

xr.use_spmd()
device = torch_xla.device()

mesh_shape = (1, 2)
device_ids = np.array(range(2))
mesh = Mesh(device_ids, mesh_shape, ("dp", "mp"))

# Test 1: Plain tensor
t1 = torch.randn(1, 32000, dtype=torch.bfloat16).to(device)
spec1 = torch_xla._XLAC._get_xla_sharding_spec(t1)
print(f"Plain tensor sharding spec: '{spec1}'")

# Test 2: After _mark_unsharded_args_replicated-style marking
from torch_xla.distributed.spmd import ShardingType

replicated = torch_xla._XLAC.OpSharding([], [], [], ShardingType.REPLICATED)
torch_xla._XLAC._xla_mark_sharding(t1, replicated)
spec2 = torch_xla._XLAC._get_xla_sharding_spec(t1)
print(f"After mark REPLICATED: '{spec2}'")

# Test 3: After torch.ops.tt.sharding_constraint
t3 = torch.randn(1, 32000, dtype=torch.bfloat16).to(device)
sdy_sharding = "#sdy.sharding_per_value<[<@mesh, [{}, {}]>]>"
t3_constrained = torch.ops.tt.sharding_constraint(t3, sdy_sharding)
spec3 = torch_xla._XLAC._get_xla_sharding_spec(t3_constrained)
print(f"After sharding_constraint: '{spec3}'")

# Test 4: After mark_sharding with replicated
t4 = torch.randn(1, 32000, dtype=torch.bfloat16).to(device)
xs.mark_sharding(t4, mesh, (None, None))
spec4 = torch_xla._XLAC._get_xla_sharding_spec(t4)
print(f"After xs.mark_sharding replicated: '{spec4}'")

# Test 5: Get specs for all as a list
specs_list = torch_xla._XLAC._get_xla_sharding_specs([t1, t3_constrained, t4])
print(f"Specs list: {specs_list}")

# Test 6: Check if _mark_unsharded_args_replicated would skip t3_constrained
needs_marking = not torch_xla._XLAC._get_xla_sharding_spec(t3_constrained)
print(f"t3_constrained needs marking (empty spec): {needs_marking}")
