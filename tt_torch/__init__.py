# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Process start method must be "forkserver" for op-by-op compilation for the following reasons:
#   - The backend will hang if attempting to retrieve the device descriptor from a forked process if there are multiple chips
#   - torch tensors cannot be made contiguous in a forked process
import multiprocessing as mp
import torch

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

# Import these modules so backends are registered ("tt", and "tt-experimental")
import tt_torch.backend.backend

from torch_xla.experimental import plugins


class TTPjrtPlugin(plugins.DevicePlugin):
    def library_path(self):
        # This is where the pjrt plugin will be located if you've built and installed from source
        direct_build_install_path = os.path.join(
            os.path.dirname(__file__), "../install/lib/pjrt_plugin_tt.so"
        )
        if os.path.exists(direct_build_install_path):
            return direct_build_install_path

        # This is where the pjrt plugin will be located if you've installed the tt-torch wheel into a virtual environment
        env_path = os.path.join(os.path.dirname(__file__), "../../../pjrt_plugin_tt.so")
        if os.path.exists(env_path):
            return env_path

        # This is where the pjrt plugin will be located if you've only built and installed the wheel - but you're running your code with the root of the source tree (CI does this)
        source_path = os.path.join(
            os.path.dirname(__file__), "../env/venv/lib/pjrt_plugin_tt.so"
        )
        if os.path.exists(source_path):
            return source_path

        tt_xla_build_path = os.path.join(
            os.path.dirname(__file__), "../build/src/tt/pjrt_plugin_tt.so"
        )
        if os.path.exists(tt_xla_build_path):
            return tt_xla_build_path

        assert False, "Could not find pjrt_plugin_tt.so"
