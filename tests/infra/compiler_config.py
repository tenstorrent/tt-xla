# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict


@dataclass
class CompilerConfig:
    """
    Configuration for compiler options passed to the TT device compilation.
    
    This class encapsulates various compiler knobs and optimizations that can be
    applied during model compilation for TT device.
    """
    
    # Enable BFP8 conversion in MLIR
    enable_bfp8_conversion: bool = False
    
    # Enable optimizer passes
    enable_optimizer: bool = False
    
    def to_jax_compiler_options(self) -> Dict[str, str]:
        """
        Convert CompilerConfig to JAX compiler_options dictionary format.
        
        Returns:
            Dictionary of compiler options in the format expected by jax.jit()
        """
        options = {}
        
        if self.enable_bfp8_conversion:
            options["bfp8"] = "true"
        
        if self.enable_optimizer:
            options["optimize"] = "true"
        
        return options