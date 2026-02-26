# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from tt_torch import parse_compiled_artifacts_from_cache_to_disk

# enable_compile_only must be called before any TT device access.
from ttxla_tools import enable_compile_only


class SimpleModel(nn.Module):
    def forward(self, x, y):
        return x + y


def main(system_desc_path: str):
    enable_compile_only(system_desc_path)

    xr.set_device_type("TT")
    cache_dir = f"{os.getcwd()}/cachedir"
    xr.initialize_cache(cache_dir)

    device = xm.xla_device()
    model = SimpleModel().to(device)
    x = torch.randn(3, 4).to(device)
    y = torch.randn(3, 4).to(device)
    output = model(x, y)

    # torch_xla uses lazy execution: compilation and execution both happen here.
    # In compile-only mode, compilation succeeds and artifacts are cached, but
    # execution raises an error since no hardware is available.
    try:
        output.to("cpu")
    except RuntimeError:
        pass  # Expected: in compile-only mode compilation is cached but execution is disabled.

    parse_compiled_artifacts_from_cache_to_disk(cache_dir, "output/model")

    print("Artifacts written to output/model.*")
    print("To run on hardware: ttrt run output/model.ttnn")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <system_desc_path>")
        sys.exit(1)

    main(sys.argv[1])
