#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path
from transformers import AutoTokenizer
import jax.tree_util as jtu

# Import both conversion functions
from convert_weights import convert_llama_weights as convert_original
from convert_weights_megatron import convert_llama_weights as convert_megatron


def print_tree_shapes(tree, name, indent=0):
    """Recursively print shapes of nested tree structure"""
    prefix = "  " * indent

    if hasattr(tree, "shape"):
        # It's a tensor/array
        print(f"{prefix}{name}: {list(tree.shape)} ({tree.dtype})")
    elif isinstance(tree, dict):
        print(f"{prefix}{name}:")
        for key, value in tree.items():
            print_tree_shapes(value, key, indent + 1)
    elif isinstance(tree, (list, tuple)):
        print(f"{prefix}{name}: [{type(tree).__name__} of {len(tree)} items]")
        for i, item in enumerate(tree):
            print_tree_shapes(item, f"[{i}]", indent + 1)
    else:
        tree_str = str(tree) if tree is not None else "None"
        print(f"{prefix}{name}: {type(tree)} = {tree_str}")


def compare_configs(config1, config2, name1, name2):
    """Compare two LLaMAConfig objects"""
    print(f"\n📊 Config Comparison: {name1} vs {name2}")
    print("=" * 60)

    # Get all attributes
    attrs1 = {k: v for k, v in config1.__dict__.items()}
    attrs2 = {k: v for k, v in config2.__dict__.items()}

    all_keys = set(attrs1.keys()) | set(attrs2.keys())

    for key in sorted(all_keys):
        val1 = attrs1.get(key, "MISSING")
        val2 = attrs2.get(key, "MISSING")

        # Convert None to string for formatting
        val1_str = str(val1) if val1 is not None else "None"
        val2_str = str(val2) if val2 is not None else "None"

        if val1 == val2:
            status = "✅"
        else:
            status = "❌"

        print(f"{status} {key:<25} | {val1_str:<15} | {val2_str:<15}")


def compare_outputs(ckpt_dir):
    """Compare outputs from both conversion methods"""

    print("🔧 Loading tokenizer...")
    model_id = "meta-llama/Meta-Llama-3.1-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print("📊 Loading weights with ORIGINAL convert_weights.py...")
    try:
        original_weights, original_config = convert_original(
            ckpt_dir=ckpt_dir,
            tokenizer=tokenizer,
            max_seq_len=512,
            n_layers=1,
            verbose=False,
        )
        print("✅ Original weights loaded successfully")
    except Exception as e:
        print(f"❌ Original loading failed: {e}")
        return

    print("\n📊 Loading weights with MEGATRON convert_weights_megatron.py...")
    try:
        # Load for rank 0 only for comparison
        megatron_weights, megatron_config = convert_megatron(
            ckpt_dir=ckpt_dir,
            tokenizer=tokenizer,
            max_seq_len=512,
            n_layers=1,
            tp_rank=0,
            verbose=False,
        )
        print("✅ Megatron weights loaded successfully")
    except Exception as e:
        print(f"❌ Megatron loading failed: {e}")
        return

    # Compare configs
    compare_configs(original_config, megatron_config, "Original", "Megatron")

    print(f"\n📏 ORIGINAL WEIGHTS STRUCTURE:")
    print("=" * 50)
    print_tree_shapes(original_weights, "weights")

    print(f"\n📏 MEGATRON WEIGHTS STRUCTURE (rank 0):")
    print("=" * 50)
    print_tree_shapes(megatron_weights, "weights")

    # Compare specific weight shapes
    print(f"\n🔍 DETAILED WEIGHT COMPARISON (Layer 0):")
    print("=" * 60)
    print(f"{'Weight':<20} {'Original Shape':<20} {'Megatron Shape':<20} {'Status'}")
    print("-" * 80)

    # Get layer 0 for comparison
    try:
        orig_layer = original_weights["transformer"]["h"]["0"]
        meg_layer = megatron_weights["transformer"]["h"]["0"]

        comparisons = [
            (
                "wte.embedding",
                original_weights["transformer"]["wte"]["embedding"],
                megatron_weights["transformer"]["wte"]["embedding"],
            ),
            (
                "ln_f.kernel",
                original_weights["transformer"]["ln_f"]["kernel"],
                megatron_weights["transformer"]["ln_f"]["kernel"],
            ),
            (
                "lm_head.kernel",
                original_weights["lm_head"]["kernel"],
                megatron_weights["lm_head"]["kernel"],
            ),
            (
                "wq.kernel",
                orig_layer["attention"]["wq"]["kernel"],
                meg_layer["attention"]["wq"]["kernel"],
            ),
            (
                "wk.kernel",
                orig_layer["attention"]["wk"]["kernel"],
                meg_layer["attention"]["wk"]["kernel"],
            ),
            (
                "wv.kernel",
                orig_layer["attention"]["wv"]["kernel"],
                meg_layer["attention"]["wv"]["kernel"],
            ),
            (
                "wo.kernel",
                orig_layer["attention"]["wo"]["kernel"],
                meg_layer["attention"]["wo"]["kernel"],
            ),
            (
                "w1.kernel",
                orig_layer["feed_forward"]["w1"]["kernel"],
                meg_layer["feed_forward"]["w1"]["kernel"],
            ),
            (
                "w2.kernel",
                orig_layer["feed_forward"]["w2"]["kernel"],
                meg_layer["feed_forward"]["w2"]["kernel"],
            ),
            (
                "w3.kernel",
                orig_layer["feed_forward"]["w3"]["kernel"],
                meg_layer["feed_forward"]["w3"]["kernel"],
            ),
            (
                "attention_norm",
                orig_layer["attention_norm"]["kernel"],
                meg_layer["attention_norm"]["kernel"],
            ),
            (
                "ffn_norm",
                orig_layer["ffn_norm"]["kernel"],
                meg_layer["ffn_norm"]["kernel"],
            ),
        ]
    except Exception as e:
        print(f"❌ Error accessing layer weights: {e}")
        return

    for name, orig_tensor, meg_tensor in comparisons:
        try:
            orig_shape = list(orig_tensor.shape)
            meg_shape = list(meg_tensor.shape)

            if orig_shape == meg_shape:
                status = "🟢 Same"
            elif len(orig_shape) == len(meg_shape) and len(orig_shape) == 2:
                # Check for column sharding (output dimension reduced)
                if orig_shape[0] == meg_shape[0] and orig_shape[1] > meg_shape[1]:
                    status = "🔵 Column Sharded"
                # Check for row sharding (input dimension reduced)
                elif orig_shape[1] == meg_shape[1] and orig_shape[0] > meg_shape[0]:
                    status = "🔵 Row Sharded"
                else:
                    status = "🔴 Different"
            else:
                status = "🔴 Different"

            print(f"{name:<20} {str(orig_shape):<20} {str(meg_shape):<20} {status}")
        except Exception as e:
            print(f"{name:<20} {'ERROR':<20} {'ERROR':<20} ❌ {e}")

    # Memory comparison
    print(f"\n💾 MEMORY USAGE COMPARISON:")
    print("=" * 40)

    def count_params(tree):
        total = 0

        def count_leaf(x):
            nonlocal total
            try:
                if hasattr(x, "size"):
                    total += x.size
            except:
                pass  # Skip problematic leaves

        jtu.tree_map(count_leaf, tree)
        return total

    orig_params = count_params(original_weights)
    meg_params = count_params(megatron_weights)

    # Calculate replicated vs sharded weights
    def count_replicated_sharded(meg_tree, orig_tree):
        replicated = 0
        sharded = 0

        def count_leaf(meg_x, orig_x, path):
            nonlocal replicated, sharded
            try:
                if hasattr(meg_x, "shape") and hasattr(orig_x, "shape"):
                    if meg_x.shape == orig_x.shape:
                        replicated += meg_x.size  # Same shape = replicated
                    else:
                        sharded += meg_x.size  # Different shape = sharded
            except:
                pass

        def traverse(meg_subtree, orig_subtree, path=""):
            if isinstance(meg_subtree, dict) and isinstance(orig_subtree, dict):
                for key in meg_subtree:
                    if key in orig_subtree:
                        traverse(meg_subtree[key], orig_subtree[key], f"{path}.{key}")
            else:
                count_leaf(meg_subtree, orig_subtree, path)

        traverse(meg_tree, orig_tree)
        return replicated, sharded

    replicated_params, sharded_params = count_replicated_sharded(
        megatron_weights, original_weights
    )

    # For sharded weights: each rank has 1/4, so original total was sharded_params * 4
    # But the actual total across ranks for sharded weights is the ORIGINAL size (not replicated)
    original_sharded_total = sharded_params * 4
    total_across_ranks = replicated_params * 4 + original_sharded_total

    print(
        f"Original weights:    {orig_params:,} parameters ({orig_params * 4 / 1e9:.2f}GB fp32)"
    )
    print(
        f"Megatron rank 0:     {meg_params:,} parameters ({meg_params * 2 / 1e9:.2f}GB fp16)"
    )
    print(
        f"  - Replicated:      {replicated_params:,} parameters ({replicated_params * 2 / 1e9:.2f}GB fp16)"
    )
    print(
        f"  - Sharded:         {sharded_params:,} parameters ({sharded_params * 2 / 1e9:.2f}GB fp16)"
    )
    print(
        f"Total across 4 ranks: {total_across_ranks:,} parameters ({total_across_ranks * 2 / 1e9:.2f}GB fp16)"
    )
    print(
        f"Memory per rank: {((orig_params - meg_params) / orig_params * 100):.1f}% reduction from original"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare convert_weights vs convert_weights_megatron outputs"
    )
    parser.add_argument(
        "--ckpt_dir", type=str, required=True, help="Path to Llama checkpoint directory"
    )
    args = parser.parse_args()

    if not Path(args.ckpt_dir).exists():
        print(f"❌ Checkpoint directory not found: {args.ckpt_dir}")
        sys.exit(1)

    try:
        compare_outputs(args.ckpt_dir)
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
        import traceback

        traceback.print_exc()
