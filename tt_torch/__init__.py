# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Process start method must be "forkserver" for op-by-op compilation for the following reasons:
#   - The backend will hang if attempting to retrieve the device descriptor from a forked process if there are multiple chips
#   - torch tensors cannot be made contiguous in a forked process
import multiprocessing as mp
import torch_xla

if mp.get_start_method() != "forkserver":
    mp.set_start_method("forkserver", force=True)

import os
import importlib.util

# find the tt-metal directory, it can either be in the venv if installed from a wheel or in the third_party source tree
package_name = "tt-metal"
spec = importlib.util.find_spec(package_name)
if spec is not None:
    tt_metal_home = os.path.abspath(spec.submodule_search_locations[0])
    os.environ["TT_METAL_HOME"] = tt_metal_home

# Import these modules so "tt" backend is registered
import tt_torch.backend.backend

torch_xla.runtime.set_device_type("TT")
