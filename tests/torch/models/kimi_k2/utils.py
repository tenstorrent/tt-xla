# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Re-export MLA cache classes from the shared utils location.

The canonical implementations live in tests/torch/models/utils/mla_cache.py
so they can be reused across MLA-based models (Kimi K2, DeepSeek, …).
"""

from tests.torch.models.utils.mla_cache import MLACache, MLAStaticLayer

__all__ = ["MLACache", "MLAStaticLayer"]
