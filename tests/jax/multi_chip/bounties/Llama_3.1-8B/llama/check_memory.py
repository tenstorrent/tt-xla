#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Memory usage checker for multi-chip LLaMA loading
"""
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import jax.numpy as jnp
import psutil
import gc
from pathlib import Path
from transformers import AutoTokenizer
from jax.sharding import Mesh
from generate_multi_chip import jax_load


def get_memory_info():
    """Get current memory usage info"""
    process = psutil.Process()
    mem_info = process.memory_info()
    virtual_mem = psutil.virtual_memory()

    return {
        "cpu_rss_gb": mem_info.rss
        / 1024**3,  # Resident Set Size (actual physical memory)
        "cpu_vms_gb": mem_info.vms / 1024**3,  # Virtual Memory Size
        "cpu_total_gb": virtual_mem.total / 1024**3,
        "cpu_available_gb": virtual_mem.available / 1024**3,
        "cpu_percent": virtual_mem.percent,
    }


def print_memory_info(stage, mem_info):
    """Print formatted memory information"""
    print(f"\n🔍 {stage}")
    print(f"   CPU RSS (actual): {mem_info['cpu_rss_gb']:.2f} GB")
    print(f"   CPU VMS (virtual): {mem_info['cpu_vms_gb']:.2f} GB")
    print(f"   CPU Available: {mem_info['cpu_available_gb']:.2f} GB")
    print(f"   CPU Usage: {mem_info['cpu_percent']:.1f}%")


def analyze_device_weights(params, mesh):
    """Analyze what weights are on each device"""
    print(f"\n🔍 DEVICE WEIGHT ANALYSIS")
    print(f"   Mesh: {mesh}")

    # Get device information
    devices = jax.devices()
    print(f"   Devices: {devices}")

    # Analyze parameter structure
    def analyze_tree(tree, path="", max_depth=3, current_depth=0):
        total_params = 0
        total_size_mb = 0

        if current_depth > max_depth:
            return total_params, total_size_mb

        if isinstance(tree, dict):
            for key, value in tree.items():
                new_path = f"{path}.{key}" if path else key
                params_count, size_mb = analyze_tree(
                    value, new_path, max_depth, current_depth + 1
                )
                total_params += params_count
                total_size_mb += size_mb
        elif hasattr(tree, "items"):  # FrozenDict
            for key, value in tree.items():
                new_path = f"{path}.{key}" if path else key
                params_count, size_mb = analyze_tree(
                    value, new_path, max_depth, current_depth + 1
                )
                total_params += params_count
                total_size_mb += size_mb
        else:
            # This is a JAX array
            if hasattr(tree, "shape") and hasattr(tree, "dtype"):
                param_count = tree.size
                size_mb = tree.nbytes / 1024**2
                total_params += param_count
                total_size_mb += size_mb

                # Check device placement
                device_info = "Unknown"
                try:
                    if hasattr(tree, "devices"):
                        device_info = f"devices={tree.devices()}"
                    elif hasattr(tree, "device"):
                        device_info = f"device={tree.device()}"
                    elif hasattr(tree, "sharding"):
                        device_info = f"sharding={tree.sharding}"
                except:
                    device_info = "CPU"

                if current_depth <= 2:  # Only print details for top-level parameters
                    print(
                        f"   📊 {path}: shape={tree.shape}, dtype={tree.dtype}, "
                        f"params={param_count:,}, size={size_mb:.1f}MB, {device_info}"
                    )

        return total_params, total_size_mb

    total_params, total_size_mb = analyze_tree(params)
    print(f"\n   📈 TOTAL: {total_params:,} parameters, {total_size_mb:.1f} MB")

    # Also analyze the structure
    print(f"\n   🏗️  PARAMETER STRUCTURE:")
    if hasattr(params, "keys"):
        for key in params.keys():
            print(f"      - {key}: {type(params[key])}")
            if hasattr(params[key], "keys"):
                for subkey in list(params[key].keys())[:3]:  # Show first 3 subkeys
                    print(f"        - {subkey}: {type(params[key][subkey])}")
                if len(params[key].keys()) > 3:
                    print(f"        - ... and {len(params[key].keys()) - 3} more")

    return total_params, total_size_mb


def main():
    print("🚀 MEMORY USAGE ANALYSIS FOR MULTI-CHIP LLAMA")
    print("=" * 60)

    # Initial memory
    initial_mem = get_memory_info()
    print_memory_info("INITIAL MEMORY", initial_mem)

    # Setup
    print(f"\n🔧 Setting up devices and tokenizer...")
    devices = jax.devices()
    mesh = Mesh(devices, axis_names=("mp",))

    model_id = "meta-llama/Meta-Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    setup_mem = get_memory_info()
    print_memory_info("AFTER SETUP", setup_mem)

    # Load model
    print(f"\n🔧 Loading LLaMA with jax_load (n_layers=2)...")
    ckpt_dir = "../llama/llama3.1-8B/8B/original"
    tokenizer_path = "../llama/llama3.1-8B/original/original/tokenizer.model"

    llama = jax_load(
        model_id=model_id,
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        mesh=mesh,
        max_seq_length=64,
        n_layers=2,
    )

    loaded_mem = get_memory_info()
    print_memory_info("AFTER LOADING MODEL", loaded_mem)

    # Analyze device weights
    print(f"\n🔧 Analyzing device weight distribution...")
    total_params, total_size_mb = analyze_device_weights(llama.params, mesh)

    # Memory deltas
    setup_delta = setup_mem["cpu_rss_gb"] - initial_mem["cpu_rss_gb"]
    loading_delta = loaded_mem["cpu_rss_gb"] - setup_mem["cpu_rss_gb"]
    total_delta = loaded_mem["cpu_rss_gb"] - initial_mem["cpu_rss_gb"]

    print(f"\n📊 MEMORY DELTAS:")
    print(f"   Setup: +{setup_delta:.2f} GB")
    print(f"   Loading: +{loading_delta:.2f} GB")
    print(f"   Total: +{total_delta:.2f} GB")

    # Force garbage collection and check memory again
    print(f"\n🔧 Running garbage collection...")
    del llama
    gc.collect()

    final_mem = get_memory_info()
    print_memory_info("AFTER CLEANUP", final_mem)

    cleanup_delta = final_mem["cpu_rss_gb"] - loaded_mem["cpu_rss_gb"]
    print(f"\n📊 CLEANUP DELTA: {cleanup_delta:.2f} GB")

    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"   Peak memory usage: {loaded_mem['cpu_rss_gb']:.2f} GB")
    print(f"   Model parameters: {total_params:,} ({total_size_mb:.1f} MB)")
    print(
        f"   Memory efficiency: {total_size_mb/1024:.2f} GB for params vs {loading_delta:.2f} GB actual"
    )


if __name__ == "__main__":
    main()
