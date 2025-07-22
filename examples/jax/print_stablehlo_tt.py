#!/usr/bin/env python3
"""
Example to print StableHLO code when using the TT backend.
This demonstrates how the computation looks when targeting Tenstorrent hardware.
"""

import jax
import jax.numpy as jnp
import numpy as np

def main():
    print("Available devices:", jax.devices())
    
    # Try to get TT devices if available
    try:
        tt_devices = jax.devices("tt")
        print(f"TT devices: {tt_devices}")
        if tt_devices:
            print(f"Using TT device: {tt_devices[0]}")
    except Exception as e:
        print(f"TT devices not available: {e}")
        print("Using default device")
    
    # Create input data matching the multihost example
    global_data = np.arange(32).reshape((4, 8)).astype(np.float32)
    array = jnp.array(global_data)
    
    print(f"Input shape: {array.shape}, dtype: {array.dtype}")
    
    # Define the computation (same as multihost example)
    def computation(x):
        """Sum of sine of input array"""
        return jnp.sum(jnp.sin(x))
    
    # Print StableHLO for different compilation strategies
    print("\n" + "="*60)
    print("STABLEHLO REPRESENTATIONS")
    print("="*60)
    
    # 1. Basic JIT compilation
    print("\n1. Basic JIT compilation:")
    print("-" * 40)
    try:
        jit_fn = jax.jit(computation)
        hlo_computation = jax.xla_computation(jit_fn)(array)
        print("HLO Text:")
        hlo_text = hlo_computation.as_hlo_text()
        print(hlo_text)
    except Exception as e:
        print(f"Error: {e}")
    
    # 2. Lower and inspect
    print("\n2. Lowered representation:")
    print("-" * 40)
    try:
        lowered = jax.jit(computation).lower(array)
        print("Compiler IR:")
        print(lowered.as_text())
        
        # Get the HLO from lowered
        print("\nHLO from lowered:")
        hlo_modules = lowered.compiler_ir()
        print(hlo_modules)
    except Exception as e:
        print(f"Error: {e}")
    
    # 3. Compile and inspect
    print("\n3. Compiled representation:")
    print("-" * 40)
    try:
        compiled = jax.jit(computation).lower(array).compile()
        print("Compiled IR:")
        print(compiled.as_text())
    except Exception as e:
        print(f"Error: {e}")
    
    # 4. Show specific to TT backend if available
    print("\n4. Backend-specific information:")
    print("-" * 40)
    try:
        # Force compilation to specific backend
        with jax.default_device(jax.devices()[0]):
            backend_lowered = jax.jit(computation).lower(array)
            print(f"Backend: {backend_lowered.compile().runtime_platform()}")
            print("Platform-specific lowered representation:")
            print(backend_lowered.as_text())
    except Exception as e:
        print(f"Error: {e}")
    
    # Execute and verify
    print("\n" + "="*60)
    print("EXECUTION")
    print("="*60)
    
    result = jax.jit(computation)(array)
    expected = np.sum(np.sin(global_data))
    
    print(f"JAX result: {result}")
    print(f"NumPy expected: {expected}")
    print(f"Difference: {abs(result - expected)}")

if __name__ == "__main__":
    main()