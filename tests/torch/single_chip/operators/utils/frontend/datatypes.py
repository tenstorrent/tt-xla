# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Frontend datatypes

from .detector import XLA_MODE


if XLA_MODE:
    from .xla.forge_stubs import Tensor
    from .xla.forge_stubs import Tensor as ForgeTensor
    from .xla.forge_stubs import Module

else:
    from forge import Tensor
    from forge import Tensor as ForgeTensor

    from forge import Module


__all__ = [
    "Tensor",
    "ForgeTensor",
    "Module",
]
