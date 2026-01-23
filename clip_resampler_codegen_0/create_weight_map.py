# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Script to create a mapping between argN.tensorbin files and PyTorch weight names.
Matches by comparing both shapes AND values.
"""

import glob
import json
import os
import sys

# Add parent dir to path to import CLIPResamplerModule
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
import ttnn

from examples.pytorch.clip_resampler_sdxl import CLIPResamplerModule  # noqa: E402

# Load the PyTorch model
MODEL_PATH = "clip_resampler_sdxl.pt"
TENSORS_DIR = "tensors"
OUTPUT_FILE = "weight_map.json"


def load_tensorbin_as_torch(filepath):
    """Load a tensorbin file and convert to torch tensor."""
    ttnn_tensor = ttnn.load_tensor(filepath)
    torch_tensor = ttnn.to_torch(ttnn_tensor)
    return torch_tensor


def find_matching_weight(tensorbin_tensor, state_dict, matched_weights):
    """Find a weight in state_dict that matches the tensorbin tensor by shape and values."""
    tensorbin_shape = tuple(tensorbin_tensor.shape)

    candidates = []
    for name, weight in state_dict.items():
        if name in matched_weights:
            continue

        # First check shape
        if tuple(weight.shape) != tensorbin_shape:
            continue

        # Then check values using allclose
        # Convert to same dtype for comparison
        weight_compare = weight.to(tensorbin_tensor.dtype)
        if torch.allclose(tensorbin_tensor, weight_compare, rtol=1e-3, atol=1e-5):
            candidates.append(name)

    if len(candidates) == 1:
        return candidates[0]
    elif len(candidates) > 1:
        print(f"  WARNING: Multiple candidates found: {candidates}")
        return candidates[0]  # Return first match
    return None


def main():
    print(f"Loading PyTorch model from {MODEL_PATH}...")
    model = torch.load(MODEL_PATH, weights_only=False)
    state_dict = model.state_dict()
    print(f"Model has {len(state_dict)} weights")

    # Get all tensorbin files
    tensorbin_files = sorted(glob.glob(os.path.join(TENSORS_DIR, "arg*.tensorbin")))
    print(f"Found {len(tensorbin_files)} tensorbin files")

    weight_map = {}
    matched_weights = set()

    for filepath in tensorbin_files:
        arg_name = os.path.basename(filepath).replace(".tensorbin", "")
        print(f"Processing {arg_name}...")

        tensorbin_tensor = load_tensorbin_as_torch(filepath)
        print(
            f"  Shape: {tuple(tensorbin_tensor.shape)}, dtype: {tensorbin_tensor.dtype}"
        )

        match = find_matching_weight(tensorbin_tensor, state_dict, matched_weights)

        if match:
            weight_map[arg_name] = match
            matched_weights.add(match)
            print(f"  Matched: {match}")
        else:
            print(f"  WARNING: No match found!")
            weight_map[arg_name] = None

    # Save the mapping
    print(f"\nSaving weight map to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(weight_map, f, indent=2)

    # Mark special tensors that are not model weights
    # arg387: position indices [0, 1, 2, ...] - generated at runtime
    # arg390: input image tensor - the actual model input
    if "arg387" in weight_map and weight_map["arg387"] is None:
        weight_map["arg387"] = "__POSITION_IDS__"
    if "arg390" in weight_map and weight_map["arg390"] is None:
        weight_map["arg390"] = "__INPUT__"

    # Summary
    matched = sum(
        1 for v in weight_map.values() if v is not None and not v.startswith("__")
    )
    special = sum(
        1 for v in weight_map.values() if v is not None and v.startswith("__")
    )
    print(f"\nMatched {matched}/{len(weight_map)} tensors to model weights")
    print(f"Special tensors (not weights): {special}")

    unmatched = [k for k, v in weight_map.items() if v is None]
    if unmatched:
        print(f"Unmatched: {unmatched}")


if __name__ == "__main__":
    main()
