# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Tuple, Union

import jax
import jax.numpy as jnp
import torch
from infra.runners import run_on_cpu
from infra.utilities import Framework, Tensor
from infra.workloads import JaxMultichipWorkload, Workload
from jax._src.typing import DTypeLike
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding


def sanitize_test_name(test_name: str) -> str:
    """
    Sanitize a test name for use in filenames by replacing special characters with underscores.

    Replaces brackets, parentheses, commas, hyphens, spaces, and forward slashes with underscores
    to create a filesystem-safe name without creating subdirectories.

    Args:
        test_name: Test name to sanitize (e.g., "test_model[param1,param2]/case1")

    Returns:
        Sanitized name safe for filenames (e.g., "test_model_param1_param2_case1")

    Examples:
        >>> sanitize_test_name("test_mnist[256-128-64]")
        'test_mnist_256_128_64'
        >>> sanitize_test_name("test_all_models/pytorch_wide_resnet50_2")
        'test_all_models_pytorch_wide_resnet50_2'
    """
    # Replace special chars (including forward slashes) with underscores
    clean_name = re.sub(r"[\[\](),\-\s/:]+", "_", test_name)
    # Remove trailing underscores
    return clean_name.rstrip("_")


def random_image(image_size: int, framework: Framework = Framework.JAX) -> Tensor:
    """Create a random input image with the given image size."""
    return random_tensor(
        (image_size, image_size, 3),
        dtype=jnp.uint8,
        minval=0,
        maxval=256,
        framework=framework,
    )


def random_tensor(
    shape: Tuple,
    dtype: str | DTypeLike | torch.dtype = "float32",
    random_seed: int = 0,
    minval: float = 0.0,
    maxval: float = 1.0,
    framework: Framework = Framework.JAX,
) -> Tensor:
    """
    Generates a random tensor of `shape`, `dtype`, and `random_seed` in range
    [`minval`, `maxval`) for the desired `framework`.
    """
    return (
        random_jax_tensor(shape, dtype, random_seed, minval, maxval)
        if framework == Framework.JAX
        else random_torch_tensor(shape, dtype, random_seed, minval, maxval)
    )


@run_on_cpu(Framework.JAX)
def random_jax_tensor(
    shape: Tuple,
    dtype: str | DTypeLike = jnp.float32,
    random_seed: int = 0,
    minval: float = 0.0,
    maxval: float = 1.0,
) -> jax.Array:
    """
    Generates a random jax tensor of `shape`, `dtype`, and `random_seed` in range
    [`minval`, `maxval`).
    """

    def _str_to_dtype(dtype_str: str):
        return jnp.dtype(dtype_str)

    dtype_converted = _str_to_dtype(dtype) if isinstance(dtype, str) else dtype

    prng_key = jax.random.PRNGKey(random_seed)

    if jnp.issubdtype(dtype_converted, jnp.integer):
        return jax.random.randint(
            key=prng_key,
            shape=shape,
            dtype=dtype_converted,
            minval=int(minval),
            maxval=int(maxval),
        )
    elif jnp.issubdtype(dtype_converted, jnp.floating):
        return jax.random.uniform(
            key=prng_key,
            shape=shape,
            dtype=dtype_converted,
            minval=minval,
            maxval=maxval,
        )
    elif jnp.issubdtype(dtype_converted, jnp.bool):
        # Generate random tensor of type bool.
        return jax.random.choice(key=prng_key, a=jnp.array([False, True]), shape=shape)
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")


@run_on_cpu(Framework.TORCH)
def random_torch_tensor(
    shape: Tuple,
    dtype: Union[str, torch.dtype] = torch.float32,
    random_seed: int = 0,
    minval: float = 0.0,
    maxval: float = 1.0,
) -> torch.Tensor:
    """
    Generates a random torch tensor of `shape`, `dtype`, and `random_seed` in range
    [`minval`, `maxval`).
    """

    def _str_to_dtype(dtype_str: str):
        return getattr(torch, dtype_str)

    dtype_converted = _str_to_dtype(dtype) if isinstance(dtype, str) else dtype

    torch.manual_seed(random_seed)

    if dtype_converted in (
        torch.float64,
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ):
        return torch.empty(shape, dtype=dtype_converted).uniform_(minval, maxval)
    elif dtype_converted in (
        torch.int32,
        torch.int64,
        torch.int16,
        torch.int8,
        torch.uint8,
    ):
        return torch.randint(int(minval), int(maxval), shape, dtype=dtype_converted)
    elif dtype_converted == torch.bool:
        return torch.randint(0, 2, shape, dtype=torch.bool)
    else:
        raise TypeError(f"Unsupported dtype: {dtype_converted}")


def create_jax_inference_tester(
    model_tester_class, variant_or_args, format: str, compiler_config=None, **kwargs
):
    """Generic JAX inference tester creator."""
    from infra.testers.compiler_config import CompilerConfig

    if format == "float32":
        dtype = jnp.float32
    elif format == "bfloat16":
        dtype = jnp.bfloat16
    elif format == "bfp8":
        dtype = jnp.bfloat16
        if compiler_config is None:
            compiler_config = CompilerConfig()
        compiler_config.enable_bfp8_conversion = True

    return model_tester_class(
        variant_or_args, compiler_config=compiler_config, dtype_override=dtype, **kwargs
    )


def create_torch_inference_tester(
    model_tester_class,
    variant_or_args,
    format: str,
    compiler_config=None,
    **kwargs,
):
    """Generic PyTorch inference tester creator."""
    from infra.testers.compiler_config import CompilerConfig

    if format == "float32":
        dtype = None
    elif format == "bfloat16":
        dtype = torch.bfloat16
    elif format == "bfp8":
        dtype = torch.bfloat16
        if compiler_config is None:
            compiler_config = CompilerConfig()
        compiler_config.enable_bfp8_conversion = True

    return model_tester_class(
        variant_or_args, compiler_config=compiler_config, dtype_override=dtype, **kwargs
    )


def compile_jax_workload_for_cpu(workload: Workload) -> None:
    """Compile JAX workload for CPU using jax.jit."""
    workload.compiled_executable = jax.jit(
        workload.executable,
        static_argnames=workload.static_argnames,
    )


def compile_jax_workload_for_tt_device(
    workload: Workload, compiler_options: dict = None
) -> None:
    """Compile JAX workload for TT device using jax.jit with compiler options."""
    workload.compiled_executable = jax.jit(
        workload.executable,
        static_argnames=workload.static_argnames,
        compiler_options=compiler_options or {},
    )


def compile_torch_workload_for_cpu(workload: Workload) -> None:
    """Compile Torch workload for CPU using inductor backend."""
    to_compile = workload.model if workload.model is not None else workload.executable
    workload.compiled_executable = torch.compile(to_compile, backend="inductor")


def compile_torch_workload_for_tt_device(workload: Workload, torch_options: dict = None) -> None:
    """Compile Torch workload for TT device using tt backend."""
    to_compile = workload.model if workload.model is not None else workload.executable
    workload.compiled_executable = torch.compile(to_compile, backend="tt", options=torch_options if torch_options is not None else {})


def compile_jax_multichip_workload(
    workload: JaxMultichipWorkload, compiler_options: dict = None
) -> None:
    """
    Compile JAX multichip workload with shard_map wrapping for distributed execution.

    Sets up workload.executable for just-in-time compile and execution.
    The workload.device_mesh defines for which device (TT or CPU) it will be compiled.
    """
    module_sharded_executable = (
        shard_map(
            workload.executable,
            mesh=workload.device_mesh,
            in_specs=workload.in_specs,
            out_specs=workload.out_spec,
        )
        if workload.sharding_mode.requires_shard_map
        else workload.executable
    )
    output_sharding = NamedSharding(workload.device_mesh, workload.out_spec)
    workload.compiled_executable = jax.jit(
        module_sharded_executable,
        out_shardings=output_sharding,
        static_argnames=workload.static_argnames,
        compiler_options=compiler_options or {},
    )
