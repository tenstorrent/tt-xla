# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Import module so "tt" backend is registered
import tt_torch.backend.backend

# Import module so custom operations are registered
import tt_torch.custom_ops

from .codegen import codegen_cpp, codegen_py
from .serialization import (
    parse_compiled_artifacts_from_cache,
    parse_compiled_artifacts_from_cache_to_disk,
)
from .tools import mark_module_user_inputs
