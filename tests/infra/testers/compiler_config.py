# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class CompilerConfig:
    """
    Configuration for compiler options passed to the TT device compilation.

    This class encapsulates various compiler knobs and optimizations that can be
    applied during model compilation for TT device.
    """

    # Optimization level (0, 1, or 2) that controls multiple optimizer passes.
    # Level 0 (default): All MLIR optimizer passes disabled
    # Level 1: Basic optimizations
    #     * Consteval prepare for conv2d weights
    #     * Remove some op workarounds
    #     * Enable conv2d + mul fusing
    #     * Op level validation for inputs/outputs
    # Level 2: Advanced optimizations
    #     * All level 1 optimizations
    #     * Memory layout optimizations (sharding)
    optimization_level: int = 0

    # Enables automatic MLIR graph conversion into block fp8 format. This is
    # supported only when the graph is in bfloat16 format, to avoid loss in precision.
    # Final graph will have input and output nodes in bfloat16 and everything
    # else in bfp8. Essentially adding type casts at the beginning and in the end
    # of the graph, while all intermediate results are in bfp8. This bfloat16
    # wrapping is done because block formats are TT hardware specific, and user
    # should provide and get tensors of common dtype.
    enable_bfp8_conversion: bool = False

    # Enables experimental BFP8 weight conversion in MLIR optimizer passes.
    experimental_enable_weight_bfp8_conversion: bool = False

    # Enables Conv2d fusion with multiply pattern in the TTNN fusing pass.
    # TODO(sdjordjevicTT): This is a temporary option and will be removed once the underlying
    # issue https://github.com/tenstorrent/tt-mlir/issues/4628 is fixed.
    experimental_enable_fusing_conv2d_with_multiply_pattern: bool = False

    # Enables trace hoisting for TTNN pipeline.
    enable_trace: bool = False

    # Enable dumping of intermediate IRs to disk.
    export_path: Optional[str] = None

    # Enable dumping of model input and parameter tensors to disk.
    export_tensors: bool = False

    def to_jax_compiler_options(self) -> Dict[str, str]:
        """
        Convert CompilerConfig to JAX compiler_options dictionary format.

        Returns:
            Dictionary of compiler options in the format expected by jax.jit()
        """
        options = {}

        if self.optimization_level:
            options["optimization_level"] = str(self.optimization_level)

        if self.enable_bfp8_conversion:
            options["enable_bfp8_conversion"] = "true"

        if self.experimental_enable_weight_bfp8_conversion:
            options["experimental_enable_weight_bfp8_conversion"] = "true"

        if self.experimental_enable_fusing_conv2d_with_multiply_pattern:
            options["experimental_enable_fusing_conv2d_with_multiply_pattern"] = "true"

        if self.enable_trace:
            options["enable_trace"] = "true"

        if self.export_path:
            options["export_path"] = self.export_path

        if self.export_tensors:
            options["export_tensors"] = "true"

        return options

    def to_torch_compile_options(self) -> Dict[str, str]:
        """
        Convert CompilerConfig to Torch compile options dictionary format.

        Returns:
            Dictionary of compiler options in the format expected by torch_xla.set_custom_compile_options()
        """

        # Currently, the options are the same as JAX. But keeping separate method
        # in case of future differences.
        return self.to_jax_compiler_options()
