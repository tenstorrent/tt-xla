#!/usr/bin/env python3
"""
Simple example to print StableHLO code from the multihost computation.
This version runs on a single process for easier debugging.
"""

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
import numpy as np

def main():
    print("Available devices:", jax.devices())
    
    # Create some toy data similar to the multihost example
    global_data = np.arange(32).reshape((4, 8))
    print(f"Input data shape: {global_data.shape}")
    
    # Create a simple array (no sharding for simplicity)
    array = jnp.array(global_data)
    
    # Define the computation function (same as multihost example)
    @jax.jit
    def computation(x):
        return jnp.sum(jnp.sin(x))
    
    print("\n=== StableHLO/HLO Code ===")
    
    # Method 1: Using jax.xla_computation to get HLO
    try:
        hlo_computation = jax.xla_computation(computation)(array)
        print("HLO representation using xla_computation:")
        print(hlo_computation.as_hlo_text())
        print("-" * 50)
    except Exception as e:
        print(f"Error with xla_computation: {e}")
    
    # Method 2: Using lower() API for more detailed representation
    try:
        lowered = jax.jit(computation).lower(array)
        print("Lowered representation:")
        print(lowered.as_text())
        print("-" * 50)
    except Exception as e:
        print(f"Error with lowered: {e}")
    
    # Method 3: Using compile() to see compiled representation
    try:
        compiled = jax.jit(computation).lower(array).compile()
        print("Compiled representation:")
        print(compiled.as_text())
        print("-" * 50)
    except Exception as e:
        print(f"Error with compiled: {e}")
    
    # Execute the computation
    print("\n=== Execution Result ===")
    result = computation(array)
    print(f"Result: {result}")
    print(f"Expected (numpy): {np.sum(np.sin(global_data))}")

if __name__ == "__main__":
    main()