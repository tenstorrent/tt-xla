# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from .jax_model_setup import (
    cast_jax_inputs_dtype,
    cast_jax_model_dtype,
    configure_jax_model,
    create_jax_model_workload,
    get_jax_forward_method_args,
    get_jax_forward_method_kwargs,
    get_jax_input_parameters,
    get_jax_static_argnames,
    run_jax_training,
)
from .torch_model_setup import (
    cast_torch_inputs_dtype,
    cast_torch_model_dtype,
    configure_torch_model,
    create_torch_model_workload,
    run_torch_training,
)
