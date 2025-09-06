# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax

NUM_VIRTUAL_DEVICES = 8
jax.config.update("jax_num_cpu_devices", NUM_VIRTUAL_DEVICES)
cpu_devices = jax.devices("cpu")
axis_name = "X"
num_devices = len(cpu_devices)
device_mesh = jax.make_mesh((num_devices,), (axis_name), devices=cpu_devices)
