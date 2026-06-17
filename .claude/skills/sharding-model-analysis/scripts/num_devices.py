# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch_xla.runtime as xr


# Get number of devices
# NOTE: first activate environment by running `source venv/activate`
def get_num_devices() -> int:
    return xr.global_runtime_device_count()


if __name__ == "__main__":
    print(f"Number of devices: {get_num_devices()}")
