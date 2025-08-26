# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from .detector import XLA_MODE

if XLA_MODE:
    from .xla.provider import XlaFrontend as SweepsFrontend
else:
    from .forge.provider import ForgeFrontend as SweepsFrontend

__all__ = [
    "XLA_MODE",
    "SweepsFrontend",
]
