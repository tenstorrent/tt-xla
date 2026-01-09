# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from .providers import RMSNormFusionProvider
from .registry import run_fusion_passes
from .utils import (
    DEFAULT_DTYPES,
    FusionPattern,
    apply_fusion_pattern,
    get_registered_providers,
    make_dtype_patterns,
    register_fusion_provider,
)

__all__ = [
    # Core types
    "FusionPattern",
    # Helper functions
    "apply_fusion_pattern",
    "make_dtype_patterns",
    "register_fusion_provider",
    "get_registered_providers",
    # Constants
    "DEFAULT_DTYPES",
    # Entry point
    "run_fusion_passes",
    # Providers
    "RMSNormFusionProvider",
]
