# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch

# Apply JAX compatibility patches globally to fix TorchDynamo errors
from .utils import apply_jax_compatibility_patches, is_torch_2_10_or_newer

# Apply patches globally
apply_jax_compatibility_patches()


def _apply_xla_dynamo_guard_repr_patch():
    if not is_torch_2_10_or_newer():
        return

    try:
        from torch._dynamo.guards import GuardBuilder, get_verbose_code_parts
        from torch._dynamo.source import LocalSource, TypeSource
        from torch._guards import Guard
    except Exception:
        return

    if getattr(GuardBuilder.id_match_unchecked, "_tt_xla_guard_repr_patch", False):
        return

    def id_match_unchecked(self, guard, recompile_hint=None) -> None:
        # PyTorch uses repr(val) only to annotate verbose guard strings.
        # For XLA tensors/parameters, repr() calls tensor.to("cpu"), which
        # materializes sharded weights via ReplicateShardedData during guard
        # construction. Replace it with a cheap metadata-only string.
        if isinstance(guard.originating_source, TypeSource):
            return self.TYPE_MATCH(
                Guard(guard.originating_source.base, GuardBuilder.TYPE_MATCH)
            )

        ref = self.arg_ref(guard)
        val = self.get(guard)
        id_val = self.id_ref(val, guard.name)
        try:
            if isinstance(val, torch.Tensor) and val.device.type == "xla":
                type_repr = f"<{type(val).__name__} device={val.device}>"
            else:
                type_repr = repr(val)
        except Exception:
            type_repr = f"<{type(val).__name__}>"

        code = f"___check_obj_id({ref}, {id_val}), type={type_repr}"
        self._set_guard_export_info(guard, [code], provided_func_name="ID_MATCH")
        self.get_guard_manager(guard).add_id_match_guard(
            id_val, get_verbose_code_parts(code, guard, recompile_hint)
        )

        if isinstance(guard.originating_source, LocalSource):
            if isinstance(val, torch.nn.Module):
                local_name = guard.originating_source.local_name
                weak_id = self.lookup_weakrefs(val)
                if weak_id is not None:
                    self.id_matched_objs[local_name] = weak_id

    id_match_unchecked._tt_xla_guard_repr_patch = True
    GuardBuilder.id_match_unchecked = id_match_unchecked


_apply_xla_dynamo_guard_repr_patch()

# Import module so "tt" backend is registered
import tt_torch.backend.backend

# Import module so custom operations are registered
import tt_torch.custom_ops

# Import torch overrides so they are registered
import tt_torch.torch_overrides
from ttxla_tools import enable_compile_only, save_system_descriptor_to_disk

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
