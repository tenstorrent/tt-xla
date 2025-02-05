# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import jax
from typing import Sequence

from .workload import Workload


@dataclass
class MultichipWorkload(Workload):
    """
    Convenience dataclass storing a callable and its positional and keyword arguments.
    """

    mesh: jax.sharding.Mesh = None
    in_specs: Sequence[jax.sharding.PartitionSpec] = None
