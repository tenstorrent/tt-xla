# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Import module so "tt" backend is registered
import tt_torch.backend.backend

# Import module so custom operations are registered
import tt_torch.custom_ops

# Import torch overrides so they are registered
import tt_torch.torch_overrides
from ttxla_tools import enable_compile_only, save_system_descriptor_to_disk

from .codegen import codegen_cpp, codegen_py

# HF transformers MoE experts backend that lowers to torch.ops.tt.sparse_matmul
from .moe_backend import (
    TT_MOE_BACKEND_NAME,
    register_tt_moe_backend,
    tt_experts_forward,
)
from .serialization import (
    parse_compiled_artifacts_from_cache,
    parse_compiled_artifacts_from_cache_to_disk,
)
from .sharding import sharding_constraint_hook

# Sparse MLP for MoE models
from .sparse_mlp import A2aSparseMLP, SparseMLP, enable_sparse_mlp
from .tools import mark_module_user_inputs
from .weight_dtype import (
    apply_weight_dtype_overrides,
    dump_weight_names,
    remove_weight_dtype_overrides,
)
