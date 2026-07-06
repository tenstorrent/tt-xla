# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Apply XLA Dynamo guard repr patch and TorchDynamo compatibility patches globally to fix TorchDynamo errors
from .utils import apply_dynamo_compatibility_patches, apply_xla_dynamo_guard_repr_patch

# Apply patches globally
apply_xla_dynamo_guard_repr_patch()
apply_dynamo_compatibility_patches()

# Import module so "tt" backend is registered
import tt_torch.backend.backend

# Import module so custom operations are registered
import tt_torch.custom_ops

# Import torch overrides so they are registered
import tt_torch.torch_overrides
from ttxla_tools import save_system_descriptor_to_disk

from .codegen import codegen_cpp, codegen_py
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

_HF_BACKEND_EXPORTS = {
    "TT_MOE_BACKEND_NAME": "moe_backend",
    "TT_DENSE_EXPERTS_BACKEND_NAME": "moe_backend",
    "get_tt_moe_shard_specs": "moe_backend",
    "register_tt_moe_backend": "moe_backend",
    "tt_experts_forward": "moe_backend",
    "tt_dense_experts_forward": "moe_backend",
}


def __getattr__(name):
    module_name = _HF_BACKEND_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value
