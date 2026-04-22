# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from .torch_workload import TorchWorkload
from .workload import Workload

try:
    from .jax_workload import JaxMultichipWorkload
except Exception:
    JaxMultichipWorkload = None
