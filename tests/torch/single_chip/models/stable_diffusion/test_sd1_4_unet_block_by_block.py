# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from infra.comparators import ComparisonConfig, PccConfig, TorchComparator


@pytest.mark.parametrize("sample_size", [8, 16, 32, 64])
def test_sd1_4_unet_block_by_block(sample_size):
    """Test each UNet block individually to isolate the L1 cache issue."""
    torch.manual_seed(42)

    print(f"\n{'='*80}")
    print(f"Starting block-by-block test with sample_size={sample_size}")
    print(f"{'='*80}\n")

    print("Loading the pipeline...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print("Pipeline loaded successfully")

    print("Extracting unet module...")
    unet_module = pipe.unet
    unet_module.eval()
    print("UNet module loaded successfully")

    # Prepare base inputs
    print(f"\nPreparing inputs with sample_size={sample_size}...")
    batch_size = 1
    seq_len = 32  # Fixed sequence length
    in_channels = unet_module.config.in_channels
    feature_dim = unet_module.config.cross_attention_dim

    sample = torch.randn(batch_size, in_channels, sample_size, sample_size)
    encoder_hidden_states = torch.randn(batch_size, seq_len, feature_dim)
    timestep = torch.tensor([200], dtype=torch.long)

    # Convert to bfloat16
    sample = sample.to(torch.bfloat16)
    encoder_hidden_states = encoder_hidden_states.to(torch.bfloat16)

    # Set device type
    print("Setting device type to TT...")
    xr.set_device_type("TT")
    device = torch_xla.device()

    # Move base inputs to device
    sample_tt = sample.to(device)
    encoder_hidden_states_tt = encoder_hidden_states.to(device)
    timestep_tt = timestep.to(device)

    # Get timestep embedding (needed for blocks)
    print("\nGetting timestep embeddings...")
    with torch.no_grad():
        t_emb = unet_module.time_proj(timestep)
        t_emb = t_emb.to(torch.bfloat16)
        emb = unet_module.time_embedding(t_emb)
        emb_tt = emb.to(device)

    print(f"Timestep embedding shape: {emb.shape}")

    # Test each block
    results = {
        "conv_in": None,
        "down_blocks": [],
        "mid_block": None,
        "up_blocks": [],
        "conv_out": None,
    }

    # 1. Test conv_in
    print(f"\n{'='*80}")
    print("Testing conv_in...")
    print(f"{'='*80}")
    try:
        compiled_conv_in = torch.compile(unet_module.conv_in, backend="tt")
        compiled_conv_in.to(device)

        with torch.no_grad():
            output = compiled_conv_in(sample_tt)

        print(f"✓ conv_in PASSED - Output shape: {output.shape}")
        results["conv_in"] = "PASSED"
        conv_in_output = output
    except Exception as e:
        print(f"✗ conv_in FAILED: {e}")
        results["conv_in"] = f"FAILED: {str(e)[:100]}"
        return results

    # 2. Test down_blocks
    current_hidden_states = conv_in_output
    down_block_res_samples = []

    for i, down_block in enumerate(unet_module.down_blocks):
        print(f"\n{'='*80}")
        print(f"Testing down_block[{i}] - {down_block.__class__.__name__}")
        print(f"Input shape: {current_hidden_states.shape}")
        print(f"{'='*80}")

        try:
            compiled_down_block = torch.compile(down_block, backend="tt")
            compiled_down_block.to(device)

            with torch.no_grad():
                # Check if block has attention (CrossAttnDownBlock2D)
                if hasattr(down_block, "attentions"):
                    output, res_samples = compiled_down_block(
                        hidden_states=current_hidden_states,
                        temb=emb_tt,
                        encoder_hidden_states=encoder_hidden_states_tt,
                    )
                else:
                    # DownBlock2D without attention
                    output, res_samples = compiled_down_block(
                        hidden_states=current_hidden_states,
                        temb=emb_tt,
                    )

            print(f"✓ down_block[{i}] PASSED - Output shape: {output.shape}")
            print(f"  Residual samples: {len(res_samples)} tensors")
            results["down_blocks"].append(f"Block {i}: PASSED")

            current_hidden_states = output
            down_block_res_samples.extend(res_samples)

        except Exception as e:
            print(f"✗ down_block[{i}] FAILED: {e}")
            results["down_blocks"].append(f"Block {i}: FAILED - {str(e)[:100]}")
            print(f"\nStopping test - down_block[{i}] failed")
            return results

    # 3. Test mid_block
    print(f"\n{'='*80}")
    print("Testing mid_block - UNetMidBlock2DCrossAttn")
    print(f"Input shape: {current_hidden_states.shape}")
    print(f"{'='*80}")

    try:
        compiled_mid_block = torch.compile(unet_module.mid_block, backend="tt")
        compiled_mid_block.to(device)

        with torch.no_grad():
            output = compiled_mid_block(
                hidden_states=current_hidden_states,
                temb=emb_tt,
                encoder_hidden_states=encoder_hidden_states_tt,
            )

        print(f"✓ mid_block PASSED - Output shape: {output.shape}")
        results["mid_block"] = "PASSED"
        current_hidden_states = output

    except Exception as e:
        print(f"✗ mid_block FAILED: {e}")
        results["mid_block"] = f"FAILED: {str(e)[:100]}"
        print(f"\nStopping test - mid_block failed")
        return results

    # 4. Test up_blocks
    for i, up_block in enumerate(unet_module.up_blocks):
        print(f"\n{'='*80}")
        print(f"Testing up_block[{i}] - {up_block.__class__.__name__}")
        print(f"Input shape: {current_hidden_states.shape}")
        print(f"{'='*80}")

        try:
            compiled_up_block = torch.compile(up_block, backend="tt")
            compiled_up_block.to(device)

            # Get residual samples for this block
            num_resnets = len(up_block.resnets)
            res_samples_for_block = down_block_res_samples[-num_resnets:]
            down_block_res_samples = down_block_res_samples[:-num_resnets]

            with torch.no_grad():
                # Check if block has attention (CrossAttnUpBlock2D)
                if hasattr(up_block, "attentions"):
                    output = compiled_up_block(
                        hidden_states=current_hidden_states,
                        res_hidden_states_tuple=res_samples_for_block,
                        temb=emb_tt,
                        encoder_hidden_states=encoder_hidden_states_tt,
                    )
                else:
                    # UpBlock2D without attention
                    output = compiled_up_block(
                        hidden_states=current_hidden_states,
                        res_hidden_states_tuple=res_samples_for_block,
                        temb=emb_tt,
                    )

            print(f"✓ up_block[{i}] PASSED - Output shape: {output.shape}")
            results["up_blocks"].append(f"Block {i}: PASSED")
            current_hidden_states = output

        except Exception as e:
            print(f"✗ up_block[{i}] FAILED: {e}")
            results["up_blocks"].append(f"Block {i}: FAILED - {str(e)[:100]}")
            print(f"\nStopping test - up_block[{i}] failed")
            return results

    # 5. Test conv_out
    print(f"\n{'='*80}")
    print("Testing conv_out...")
    print(f"{'='*80}")

    try:
        # Apply normalization and activation first
        current_hidden_states = unet_module.conv_norm_out(current_hidden_states)
        current_hidden_states = unet_module.conv_act(current_hidden_states)

        compiled_conv_out = torch.compile(unet_module.conv_out, backend="tt")
        compiled_conv_out.to(device)

        with torch.no_grad():
            output = compiled_conv_out(current_hidden_states)

        print(f"✓ conv_out PASSED - Output shape: {output.shape}")
        results["conv_out"] = "PASSED"

    except Exception as e:
        print(f"✗ conv_out FAILED: {e}")
        results["conv_out"] = f"FAILED: {str(e)[:100]}"
        return results

    # Print summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"conv_in: {results['conv_in']}")
    for i, result in enumerate(results["down_blocks"]):
        print(f"down_block[{i}]: {result}")
    print(f"mid_block: {results['mid_block']}")
    for i, result in enumerate(results["up_blocks"]):
        print(f"up_block[{i}]: {result}")
    print(f"conv_out: {results['conv_out']}")
    print(f"{'='*80}\n")

    return results


if __name__ == "__main__":
    # Run with different sample sizes
    for size in [8, 16, 32]:
        print(f"\n\n{'#'*80}")
        print(f"# Running test with sample_size={size}")
        print(f"{'#'*80}\n")
        results = test_sd1_4_unet_block_by_block(size)
        if all(
            r == "PASSED"
            for r in [results["conv_in"], results["mid_block"], results["conv_out"]]
            + results["down_blocks"]
            + results["up_blocks"]
        ):
            print(f"\n✓✓✓ ALL BLOCKS PASSED WITH sample_size={size} ✓✓✓\n")
            break
        else:
            print(f"\n✗✗✗ SOME BLOCKS FAILED WITH sample_size={size} ✗✗✗\n")
