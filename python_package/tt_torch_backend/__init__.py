# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Import module so "tt" backend is registered
import tt_torch_backend.backend.backend

# Import module so custom operations are registered
import tt_torch_backend.custom_ops

from .tools import mark_module_user_inputs
from .serialization import (
    parse_compiled_artifacts_from_cache,
    parse_compiled_artifacts_from_cache_to_disk,
)
