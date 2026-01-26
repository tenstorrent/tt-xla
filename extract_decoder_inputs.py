# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch

from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant

# Initialize loader
loader = ModelLoader(variant=ModelVariant.GPT_OSS_120B, num_layers=None)

# Load model and inputs
model = loader.load_model(dtype_override=torch.bfloat16)
inputs = loader.load_inputs()

print(f"Input IDs shape: {inputs['input_ids'].shape}")
print(f"Attention mask shape: {inputs['attention_mask'].shape}")

# Hook to capture first layer inputs
captured_inputs = {}


def capture_hook(module, args, kwargs):
    """Capture inputs to the first decoder layer"""
    if not captured_inputs:  # Only capture once (first layer)
        captured_inputs["hidden_states"] = args[0].detach().clone()
        captured_inputs["attention_mask"] = kwargs.get("attention_mask")
        captured_inputs["position_ids"] = kwargs.get("position_ids")
        captured_inputs["past_key_values"] = kwargs.get("past_key_values")
        captured_inputs["use_cache"] = kwargs.get("use_cache")
        captured_inputs["cache_position"] = kwargs.get("cache_position")
        captured_inputs["position_embeddings"] = kwargs.get("position_embeddings")

        # Handle attention mask (could be dict or tensor)
        if isinstance(captured_inputs["attention_mask"], dict):
            captured_inputs["attention_mask"] = {
                k: v.detach().clone() if isinstance(v, torch.Tensor) else v
                for k, v in captured_inputs["attention_mask"].items()
            }
        elif captured_inputs["attention_mask"] is not None:
            captured_inputs["attention_mask"] = (
                captured_inputs["attention_mask"].detach().clone()
            )

        # Handle position_ids
        if captured_inputs["position_ids"] is not None:
            captured_inputs["position_ids"] = (
                captured_inputs["position_ids"].detach().clone()
            )

        # Handle cache_position
        if captured_inputs["cache_position"] is not None:
            captured_inputs["cache_position"] = (
                captured_inputs["cache_position"].detach().clone()
            )

        # Handle position_embeddings (tuple of cos, sin tensors)
        if captured_inputs["position_embeddings"] is not None:
            cos, sin = captured_inputs["position_embeddings"]
            captured_inputs["position_embeddings"] = (
                cos.detach().clone(),
                sin.detach().clone(),
            )


# Register hook on first layer
handle = model.model.layers[0].register_forward_pre_hook(capture_hook, with_kwargs=True)

# Run forward pass
with torch.no_grad():
    outputs = model(**inputs)

# Remove hook
handle.remove()

# Save captured inputs
torch.save(captured_inputs, "first_layer_inputs.pt")
print("\nSaved first layer inputs to 'first_layer_inputs.pt'")
print(f"Hidden states shape: {captured_inputs['hidden_states'].shape}")
print(f"Hidden states dtype: {captured_inputs['hidden_states'].dtype}")
