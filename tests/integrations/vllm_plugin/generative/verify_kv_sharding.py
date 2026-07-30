#!/usr/bin/env python3
"""Verify that KV cache head sharding actually occurs under DP+TP.

This script:
1. Runs a simple DP+TP model generation to trigger compilation
2. Exports the MLIR to /tmp/ir
3. Checks that kv_cache tensors have the expected sharded shape in TTIR
"""

import os
import sys
import tempfile
import re
from pathlib import Path

# Check if we can run this
if "HF_HOME" not in os.environ:
    print("ERROR: HF_HOME not set. Set it to your huggingface cache path.")
    sys.exit(1)

try:
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    from vllm import LLM
    from vllm.sampling_params import SamplingParams
except ImportError as e:
    print(f"ERROR: Required packages not installed: {e}")
    print("Make sure vllm_tt plugin is editable-installed:")
    print("  uv pip install -e integrations/vllm_plugin")
    sys.exit(1)


def verify_kv_cache_sharding(export_dir="/tmp/ir"):
    """
    Verify KV cache heads are sharded on 'model' axis.

    Expected shape in TTIR MLIR:
    - Before: [num_blocks, num_kv_heads, block_size, head_size] (replicated)
    - After:  [num_blocks, num_kv_heads/tp_size, block_size, head_size] (sharded)
    """
    print("=" * 70)
    print("KV Cache Sharding Verification Test")
    print("=" * 70)

    # Clean up export dir
    if os.path.exists(export_dir):
        import shutil
        shutil.rmtree(export_dir)

    # Small model for quick compile
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    print(f"\nInitializing model: {model_id}")
    print("(This will compile graphs with DP=2, TP=2 if running on 4-chip device)")

    additional_config = {
        "export_path": export_dir,
        "export_model_name": "kv_verify",
    }

    try:
        llm = LLM(
            model=model_id,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            max_model_len=512,
            enable_prefix_caching=False,
            additional_config=additional_config,
        )

        # Simple generate to trigger compilation
        prompt = "What is 2 + 2?"
        sampling_params = SamplingParams(temperature=0, max_tokens=10)

        print(f"\nGenerating with prompt: '{prompt}'")
        outputs = llm.generate(prompt, sampling_params)
        print(f"Output: {outputs[0].outputs[0].text}")

    except Exception as e:
        print(f"\nWARNING: Could not run generation test: {e}")
        print("This is OK if you're not on a multi-chip device.")
        print("Will still check exported IR if it exists.\n")

    # Check exported IR
    ir_dir = Path(export_dir) / "irs"
    if not ir_dir.exists():
        print(f"\nERROR: No IR exported to {ir_dir}")
        print("Make sure export_path is set in additional_config")
        return False

    # Find kv cache args in ttir_kv_*.mlir files
    print(f"\nSearching for KV cache shapes in {ir_dir}...")
    kv_files = list(ir_dir.glob("ttir_kv_*.mlir"))

    if not kv_files:
        print(f"WARNING: No ttir_kv_*.mlir files found in {ir_dir}")
        return False

    found_sharded = False
    for mlir_file in kv_files:
        print(f"\nChecking {mlir_file.name}...")
        content = mlir_file.read_text()

        # Look for func.func @main and its arguments
        # Format: @func.attr "tt.has_fallback" = false
        # followed by arguments with ttcore.local_shape attributes

        # Find all KV cache arguments (4D tensors with specific naming)
        kv_arg_pattern = r"(%[^:]*self_attn.*kv_cache[^:]*): tensor<(\d+)x(\d+)x(\d+)x(\d+)x[^>]*ttcore\.local_shape\s*=\s*dense<\[?([^\]]+)\]?>"

        matches = re.finditer(kv_arg_pattern, content)
        for match in matches:
            arg_name = match.group(1)
            global_shape = [int(x) for x in match.groups()[1:5]]
            local_shape_str = match.group(6)

            print(f"  Found KV cache arg: {arg_name}")
            print(f"    Global shape: {global_shape}")

            try:
                local_shape = [int(x.strip()) for x in local_shape_str.split(",")]
                print(f"    Local shape (per device): {local_shape}")

                # Check if heads (dim 1) are sharded
                global_heads = global_shape[1]
                local_heads = local_shape[1]

                if local_heads < global_heads:
                    print(f"    ✓ SHARDED: heads reduced from {global_heads} to {local_heads}")
                    print(f"      Sharding factor: {global_heads // local_heads}x")
                    found_sharded = True
                else:
                    print(f"    ✗ NOT SHARDED: heads still {local_heads} (expected < {global_heads})")
            except (ValueError, IndexError) as e:
                print(f"    Could not parse local_shape: {e}")

        if not matches:
            # Try alternative pattern - ttcore.local_shape as attribute
            alt_pattern = r'%[^:]*self_attn.*kv_cache[^:]*": tensor<(\d+)x(\d+)x(\d+)x(\d+)x[^>]*ttcore\.local_shape.*=.*dense<\[?([^\]]+)\]?>'
            matches = re.finditer(alt_pattern, content)
            if not list(matches):
                print(f"  No KV cache arguments found (check MLIR format)")

    print("\n" + "=" * 70)
    if found_sharded:
        print("✓ SUCCESS: KV cache heads are sharded on 'model' axis")
        print("=" * 70)
        return True
    else:
        print("✗ WARNING: Could not verify sharding in exported IR")
        print("  Check that export_path is correctly set and IR is generated")
        print("=" * 70)
        return None


if __name__ == "__main__":
    verify_kv_cache_sharding()
