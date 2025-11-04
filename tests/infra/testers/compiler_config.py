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

    # Enables optimizer passes in MLIR. This includes various optimizations
    # such as improving tensor memory layouts, operation configurations etc.
    enable_optimizer: bool = False

    # Enables memory layout analysis to allow sharded memory layouts in optimizer passes.
    enable_memory_layout_analysis: bool = False

    # Enables L1 interleaved fallback analysis in optimizer passes.
    # This analysis attempts to move tensors from DRAM to L1 memory with
    # interleaved layout when beneficial for performance.
    enable_l1_interleaved: bool = False

    # Enables automatic MLIR graph conversion into block fp8 format. This is
    # supported only when the graph is in bfloat16 format, to avoid loss in precision.
    # Final graph will have input and output nodes in bfloat16 and everything
    # else in bfp8. Essentially adding type casts at the beginning and in the end
    # of the graph, while all intermediate results are in bfp8. This bfloat16
    # wrapping is done because block formats are TT hardware specific, and user
    # should provide and get tensors of common dtype.
    enable_bfp8_conversion: bool = False

    # Enables Conv2d fusion with multiply pattern in the TTNN fusing pass.
    # TODO(sdjordjevicTT): This is a temporary option and will be removed once the underlying
    # issue https://github.com/tenstorrent/tt-mlir/issues/4628 is fixed.
    enable_fusing_conv2d_with_multiply_pattern: bool = False

    # Enables trace hoisting for TTNN pipeline.
    enable_trace: bool = False

    # Enables dumping of graph inputs to disk.
    dump_inputs: bool = False

    # Export path for generated code.
    export_path: Optional[str] = None

    def to_jax_compiler_options(self) -> Dict[str, str]:
        """
        Convert CompilerConfig to JAX compiler_options dictionary format.

        Returns:
            Dictionary of compiler options in the format expected by jax.jit()
        """
        options = {}

        if self.enable_optimizer:
            options["enable_optimizer"] = "true"

        if self.enable_memory_layout_analysis:
            options["enable_memory_layout_analysis"] = "true"

        if self.enable_l1_interleaved:
            options["enable_l1_interleaved"] = "true"

        if self.enable_bfp8_conversion:
            options["enable_bfp8_conversion"] = "true"

        if self.enable_fusing_conv2d_with_multiply_pattern:
            options["enable_fusing_conv2d_with_multiply_pattern"] = "true"

        if self.enable_trace:
            options["enable_trace"] = "true"

        if self.dump_inputs:
            options["dump_inputs"] = "true"

        if self.export_path:
            options["export_path"] = self.export_path

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
