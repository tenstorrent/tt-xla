
import torch
import torch.nn as nn
from third_party.tt_forge_models.mlp_mixer.pytorch import ModelLoader, ModelVariant
import pytest
import os

def save_model_inputs_in_ttir_order():
    """
    Extract and save all model inputs (weights, biases, and input tensors) in TTIR order.
    
    TTIR argument order (from mlp_mixer_sanity_ttir.mlir):
    0:  arg0  - l__self___norm_bias (768xbf16)
    1:  arg1  - l__self___norm_weight (768xbf16)
    2:  arg2  - l__self___blocks_11_mlp_channels_fc2_bias (768xbf16)
    3:  arg3  - l__self___blocks_11_mlp_channels_fc2_weight (768x3072xbf16)
    4:  arg4  - l__self___blocks_11_mlp_channels_fc1_bias (3072xbf16)
    5:  arg5  - l__self___blocks_11_mlp_channels_fc1_weight (3072x768xbf16)
    6:  arg6  - l__self___blocks_11_norm2_bias (768xbf16)
    7:  arg7  - l__self___blocks_11_norm2_weight (768xbf16)
    8:  arg8  - l__self___blocks_11_mlp_tokens_fc2_bias (196xbf16)
    9:  arg9  - l__self___blocks_11_mlp_tokens_fc2_weight (196x384xbf16)
    10: arg10 - l__self___blocks_11_mlp_tokens_fc1_bias (384xbf16)
    11: arg11 - l__self___blocks_11_mlp_tokens_fc1_weight (384x196xbf16)
    12: arg12 - l__self___blocks_11_norm1_bias (768xbf16)
    13: arg13 - l__self___blocks_11_norm1_weight (768xbf16)
    14: arg14 - args_1 (1x196x768xbf16) - input tensor
    15: arg15 - args_0 (1x196x768xbf16) - input tensor
    """
    
    # Create output directory
    save_dir = "mlp_sanity_inputs"
    os.makedirs(save_dir, exist_ok=True)
    
    # Load the model
    print("Loading model...")
    loader = ModelLoader(ModelVariant.MIXER_B16_224_GOOG_IN21K)
    wrapped_model = loader.load_model(dtype_override=torch.bfloat16)
    inputs_dict = loader.load_inputs(dtype_override=torch.bfloat16)
    
    # Extract the underlying model components
    blocks_11 = wrapped_model.blocks_11
    norm = wrapped_model.norm
    
    # Extract input tensors
    x1 = inputs_dict["x1"]  # args_1
    x2 = inputs_dict["x2"]  # args_0
    
    print(f"\nModel structure:")
    print(f"  blocks_11: {blocks_11}")
    print(f"  norm: {norm}")
    print(f"\nInput tensors:")
    print(f"  x1 (args_1): {x1.shape}, dtype={x1.dtype}")
    print(f"  x2 (args_0): {x2.shape}, dtype={x2.dtype}")
    
    # Save tensors in TTIR argument order
    tensors_to_save = [
        # 0: norm.bias
        (norm.bias, "l__self___norm_bias"),
        # 1: norm.weight
        (norm.weight, "l__self___norm_weight"),
        # 2: blocks_11.mlp_channels.fc2.bias
        (blocks_11.mlp_channels.fc2.bias, "l__self___blocks_11_mlp_channels_fc2_bias"),
        # 3: blocks_11.mlp_channels.fc2.weight
        (blocks_11.mlp_channels.fc2.weight, "l__self___blocks_11_mlp_channels_fc2_weight"),
        # 4: blocks_11.mlp_channels.fc1.bias
        (blocks_11.mlp_channels.fc1.bias, "l__self___blocks_11_mlp_channels_fc1_bias"),
        # 5: blocks_11.mlp_channels.fc1.weight
        (blocks_11.mlp_channels.fc1.weight, "l__self___blocks_11_mlp_channels_fc1_weight"),
        # 6: blocks_11.norm2.bias
        (blocks_11.norm2.bias, "l__self___blocks_11_norm2_bias"),
        # 7: blocks_11.norm2.weight
        (blocks_11.norm2.weight, "l__self___blocks_11_norm2_weight"),
        # 8: blocks_11.mlp_tokens.fc2.bias
        (blocks_11.mlp_tokens.fc2.bias, "l__self___blocks_11_mlp_tokens_fc2_bias"),
        # 9: blocks_11.mlp_tokens.fc2.weight
        (blocks_11.mlp_tokens.fc2.weight, "l__self___blocks_11_mlp_tokens_fc2_weight"),
        # 10: blocks_11.mlp_tokens.fc1.bias
        (blocks_11.mlp_tokens.fc1.bias, "l__self___blocks_11_mlp_tokens_fc1_bias"),
        # 11: blocks_11.mlp_tokens.fc1.weight
        (blocks_11.mlp_tokens.fc1.weight, "l__self___blocks_11_mlp_tokens_fc1_weight"),
        # 12: blocks_11.norm1.bias
        (blocks_11.norm1.bias, "l__self___blocks_11_norm1_bias"),
        # 13: blocks_11.norm1.weight
        (blocks_11.norm1.weight, "l__self___blocks_11_norm1_weight"),
        # 14: args_1 (x1)
        (x1, "args_1"),
        # 15: args_0 (x2)
        (x2, "args_0"),
    ]
    
    print(f"\nSaving {len(tensors_to_save)} tensors to {save_dir}/")
    print("-" * 80)
    
    for idx, (tensor, name) in enumerate(tensors_to_save):
        filename = f"{save_dir}/{idx}.pt"
        torch.save(tensor.detach().cpu(), filename)
        print(f"  {idx:2d}.pt: {name:50s} {str(tensor.shape):20s} dtype={tensor.dtype}")
    
    print("-" * 80)
    print(f"\nAll tensors saved successfully to {save_dir}/\n")
    
    return wrapped_model, x1, x2, save_dir


class Mlp(nn.Module):
    """MLP block as used in MLP-Mixer."""
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU(approximate='none')
        self.fc2 = nn.Linear(hidden_features, in_features, bias=True)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class MixerBlock(nn.Module):
    """Single Mixer block."""
    def __init__(self, dim=768, seq_len=196, mlp_ratio_tokens=2, mlp_ratio_channels=4):
        super().__init__()
        tokens_dim = int(seq_len * mlp_ratio_tokens)
        channels_dim = int(dim * mlp_ratio_channels)
        
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp_tokens = Mlp(seq_len, tokens_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp_channels = Mlp(dim, channels_dim)
    
    def forward(self, x):
        # Token mixing
        y = self.norm1(x)
        y = y.transpose(1, 2)  # (B, C, N)
        y = self.mlp_tokens(y)
        y = y.transpose(1, 2)  # (B, N, C)
        x = x + y
        
        # Channel mixing
        y = self.norm2(x)
        y = self.mlp_channels(y)
        x = x + y
        
        return x


class CustomWrapper(nn.Module):
    """Custom wrapper matching the original Wrapper architecture."""
    def __init__(self):
        super().__init__()
        self.blocks_11 = MixerBlock(dim=768, seq_len=196, mlp_ratio_tokens=2, mlp_ratio_channels=4)
        self.norm = nn.LayerNorm(768, eps=1e-6)
    
    def forward(self, x1, x2):
        x = x1 + x2
        x = self.blocks_11(x)
        x = self.norm(x)
        return x


def load_saved_inputs_into_model(model, save_dir):
    """Load saved tensors into the custom model."""
    
    # Load parameters
    model.norm.bias.data = torch.load(f"{save_dir}/0.pt")
    model.norm.weight.data = torch.load(f"{save_dir}/1.pt")
    model.blocks_11.mlp_channels.fc2.bias.data = torch.load(f"{save_dir}/2.pt")
    model.blocks_11.mlp_channels.fc2.weight.data = torch.load(f"{save_dir}/3.pt")
    model.blocks_11.mlp_channels.fc1.bias.data = torch.load(f"{save_dir}/4.pt")
    model.blocks_11.mlp_channels.fc1.weight.data = torch.load(f"{save_dir}/5.pt")
    model.blocks_11.norm2.bias.data = torch.load(f"{save_dir}/6.pt")
    model.blocks_11.norm2.weight.data = torch.load(f"{save_dir}/7.pt")
    model.blocks_11.mlp_tokens.fc2.bias.data = torch.load(f"{save_dir}/8.pt")
    model.blocks_11.mlp_tokens.fc2.weight.data = torch.load(f"{save_dir}/9.pt")
    model.blocks_11.mlp_tokens.fc1.bias.data = torch.load(f"{save_dir}/10.pt")
    model.blocks_11.mlp_tokens.fc1.weight.data = torch.load(f"{save_dir}/11.pt")
    model.blocks_11.norm1.bias.data = torch.load(f"{save_dir}/12.pt")
    model.blocks_11.norm1.weight.data = torch.load(f"{save_dir}/13.pt")
    
    # Load input tensors
    x1 = torch.load(f"{save_dir}/14.pt")
    x2 = torch.load(f"{save_dir}/15.pt")
    
    return model, x1, x2


def test_save_and_verify_inputs():
    """Test that saves model inputs and verifies they are correct by comparing outputs."""
    
    print("=" * 80)
    print("TEST: Save and verify MLP Mixer model inputs")
    print("=" * 80)
    
    # Step 1: Save inputs from original model
    print("\n[1/4] Saving inputs from original model...")
    original_model, x1_orig, x2_orig, save_dir = save_model_inputs_in_ttir_order()
    
    # Step 2: Run original model
    print("\n[2/4] Running original model...")
    with torch.no_grad():
        output_original = original_model(x1_orig, x2_orig)
    print(f"  Original output shape: {output_original.shape}, dtype={output_original.dtype}")
    
    # Step 3: Create custom model and load saved inputs
    print("\n[3/4] Creating custom model and loading saved inputs...")
    custom_model = CustomWrapper()
    custom_model.eval()
    custom_model, x1_loaded, x2_loaded = load_saved_inputs_into_model(custom_model, save_dir)
    print("  Custom model created and loaded successfully")
    
    # Step 4: Run custom model and compare outputs
    print("\n[4/4] Running custom model and comparing outputs...")
    with torch.no_grad():
        output_custom = custom_model(x1_loaded, x2_loaded)
    print(f"  Custom output shape: {output_custom.shape}, dtype={output_custom.dtype}")
    
    # Compare outputs
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS:")
    print("=" * 80)
    
    # Check shapes match
    shapes_match = output_original.shape == output_custom.shape
    print(f"  Shapes match: {shapes_match}")
    if shapes_match:
        print(f"    Shape: {output_original.shape}")
    else:
        print(f"    Original: {output_original.shape}, Custom: {output_custom.shape}")
    
    assert shapes_match, "Output shapes do not match!"

    print("output_original=",output_original)
    print("output_custom=",output_custom)
    
    # Check allclose
    outputs_match = torch.allclose(output_original, output_custom)
    print(f"  Outputs match (allclose ): {outputs_match}")
    

    if outputs_match:
        print("\n" + "✓" * 80)
        print("SUCCESS: Saved inputs are correct! Models produce identical outputs.")
        print("✓" * 80)
    else:
        print("\n" + "⚠" * 80)
        print("WARNING: Outputs differ beyond tolerance threshold.")
        print("This may be due to numerical precision differences.")
        print("⚠" * 80)
    
    print(f"\nAll files saved in: {save_dir}/")
    print("=" * 80)
    
    # Assert that outputs match
    assert outputs_match, "Outputs do not match within tolerance!"


if __name__ == "__main__":
    test_save_and_verify_inputs()

       