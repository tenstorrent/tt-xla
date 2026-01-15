# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import run_graph_test
from infra.comparators.comparison_config import ComparisonConfig, PccConfig
from infra.utilities import Framework
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.qwen_2_5_vl.pytorch.loader import ModelLoader


@pytest.mark.parametrize("tp_bool", [True, False])
def test_qwen_2_5_vl_language_layer(tp_bool):
    """Test Qwen2.5-VL single language decoder layer with tensor parallel sharding."""
    # Load model
    loader = ModelLoader()
    model = loader.load_model(dtype_override=torch.bfloat16)
    
    # Extract language layer
    layer = model.model.language_model.layers[0]

    print(f"\n{'='*60}")
    print(f"Testing Language Layer with tp={tp_bool}")
    print(f"{'='*60}\n")

    # Load inputs
    inputs_dict = loader.load_inputs(dtype_override=torch.bfloat16)
    
    # Mock inputs for language layer
    config = model.model.config
    text_config = config.text_config if hasattr(config, "text_config") else config
    
    batch_size = 1
    seq_len = inputs_dict["input_ids"].shape[1]
    hidden_size = text_config.hidden_size
    
    # Hidden states
    hidden_states = torch.randn(
        batch_size, seq_len, hidden_size, dtype=torch.bfloat16
    )
    # Attention mask (causal)
    attention_mask = torch.tril(torch.ones(batch_size, 1, seq_len, seq_len, dtype=torch.bfloat16))
    # Position IDs
    position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    
    # Forward signature: (hidden_states, attention_mask=None, position_ids=None, ...)
    inputs = [
        hidden_states,
        attention_mask,
        position_ids,
        None, False, False # past_key_value, output_attentions, use_cache
    ]

    xr.set_device_type("TT")

    # Setup Mesh
    mesh = None
    get_shard_spec = None
    if tp_bool:
        num_devices = xr.global_runtime_device_count()
        if num_devices < 2:
            pytest.skip("Need at least 2 devices for TP test")
        
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(layer_model, args, kwargs):
            shard_specs = {}
            # Attention
            shard_specs[layer_model.self_attn.q_proj.weight] = ("model", "batch")
            shard_specs[layer_model.self_attn.k_proj.weight] = ("model", "batch")
            shard_specs[layer_model.self_attn.v_proj.weight] = ("model", "batch")
            shard_specs[layer_model.self_attn.o_proj.weight] = ("batch", "model")
            
            for proj in [layer_model.self_attn.q_proj, layer_model.self_attn.k_proj, layer_model.self_attn.v_proj]:
                if proj.bias is not None: shard_specs[proj.bias] = ("model",)
            
            # MLP
            shard_specs[layer_model.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer_model.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer_model.mlp.down_proj.weight] = ("batch", "model")
            
            return shard_specs

    run_graph_test(
        layer,
        inputs,
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.98)),
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )

@pytest.mark.parametrize("tp_bool", [True, False])
def test_qwen_2_5_vl_vision_block(tp_bool):
    """Test Qwen2.5-VL single vision transformer block with tensor parallel sharding."""
    # Load model
    loader = ModelLoader()
    model = loader.load_model(dtype_override=torch.bfloat16)
    
    # Extract vision block
    # Structure: model.model.visual.blocks[0]
    visual_block = model.model.visual.blocks[0]

    print(f"\n{'='*60}")
    print(f"Testing Vision Block with tp={tp_bool}")
    print(f"{'='*60}\n")

    # Determine shapes from visual config
    visual_config = model.model.visual.config
    hidden_size = visual_config.embed_dim
    
    # Mock inputs for vision block
    # Forward signature: (hidden_states, grid_thw_rotary_pos_emb, attention_mask=None)
    # Note: Qwen2.5-VL vision blocks use rotary embeddings passed as 2nd arg
    
    # Mock hidden states (tokens)
    num_patches = 256 # Arbitrary number of patches
    hidden_states = torch.randn(num_patches, hidden_size, dtype=torch.bfloat16)

    # Mock rotary pos emb (rotary_pos_emb)
    # Depending on implementation, this might be a tuple of cos/sin
    head_dim = hidden_size // visual_config.num_heads
    rotary_pos_emb = torch.randn(num_patches, head_dim, dtype=torch.bfloat16)
    
    # Attention mask is optional or handled internally via cu_seqlens in actual flash attn, 
    # but for standard forward it might be None or a mask.
    # Qwen2.5 VL blocks often take just (hidden_states, rotary_pos_emb)
    
    inputs = [hidden_states, rotary_pos_emb]

    xr.set_device_type("TT")

    # Setup Mesh
    mesh = None
    get_shard_spec = None
    if tp_bool:
        num_devices = xr.global_runtime_device_count()
        if num_devices < 2:
            pytest.skip("Need at least 2 devices for TP test")
        
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(layer_model, args, kwargs):
            shard_specs = {}
            # Vision Block Structure:
            # attn.qkv (Linear), attn.proj (Linear)
            # mlp.gate_proj, mlp.up_proj, mlp.down_proj
            
            # Attention
            shard_specs[layer_model.attn.qkv.weight] = ("model", "batch")
            if layer_model.attn.qkv.bias is not None:
                shard_specs[layer_model.attn.qkv.bias] = ("model",)
            
            shard_specs[layer_model.attn.proj.weight] = ("batch", "model")
            
            # MLP
            shard_specs[layer_model.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer_model.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer_model.mlp.down_proj.weight] = ("batch", "model")
            
            return shard_specs

    run_graph_test(
        visual_block,
        inputs,
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.98)),
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )

@pytest.mark.parametrize("tp_bool", [True, False])
def test_qwen_2_5_vl_patch_embed(tp_bool):
    """Test Qwen2.5-VL patch embedding (Conv3d) with tensor parallel sharding."""
    # Load model
    loader = ModelLoader()
    model = loader.load_model(dtype_override=torch.bfloat16)
    
    # Extract patch_embed layer
    # Structure: model.model.visual.patch_embed
    patch_embed = model.model.visual.patch_embed

    print(f"\n{'='*60}")
    print(f"Testing Patch Embed with tp={tp_bool}")
    print(f"{'='*60}\n")

    # Construct input for Conv3d(3, 1280, kernel_size=(2, 14, 14), ...)
    # Input shape expected: (Batch, C, D, H, W)
    # C=3. 
    # Kernel depth=2, stride depth=2. So D must be at least 2.
    # Kernel H/W=14, stride=14. H, W must be at least 14.
    
    batch_size = 1
    C = 3
    D = 2  # Temporal dimension (or depth)
    H = 28 # 2 patches high
    W = 28 # 2 patches wide
    
    pixel_values = torch.randn(batch_size, C, D, H, W, dtype=torch.bfloat16)
    
    # Forward signature: patch_embed(pixel_values)
    inputs = [pixel_values]

    xr.set_device_type("TT")

    # Setup Mesh
    mesh = None
    get_shard_spec = None
    if tp_bool:
        num_devices = xr.global_runtime_device_count()
        if num_devices < 2:
            pytest.skip("Need at least 2 devices for TP test")
        
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(layer_model, args, kwargs):
            shard_specs = {}
            # Shard Conv3d output channels (1280)
            # visual.patch_embed.proj is the Conv3d layer
            shard_specs[layer_model.proj.weight] = ("model", "batch")
            if layer_model.proj.bias is not None:
                shard_specs[layer_model.proj.bias] = ("model",)
            
            return shard_specs

    run_graph_test(
        patch_embed,
        inputs,
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.98)),
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )

def load_config(self):
    """Load and return the configuration for the Qwen 2.5 VL model variant."""
    self.config = AutoConfig.from_pretrained(
        self._variant_config.pretrained_model_name
    )
    return self.config

def get_mesh_config(self, num_devices: int):
    if not hasattr(self, "config") or self.config is None:
        self.load_config()

    # Handle Qwen2.5-VL config structure (heads count is often in text_config)
    num_heads = getattr(self.config, "num_attention_heads", 0)
    if num_heads == 0 and hasattr(self.config, "text_config"):
        num_heads = getattr(self.config.text_config, "num_attention_heads", 0)

    # Default fallback if heads cannot be determined
    if num_heads == 0:
        return (1, num_devices), ("batch", "model")

    # Prefer (1, N) when heads divide N, otherwise try (2, N/2)
    if num_heads % num_devices == 0:
        mesh_shape = (1, num_devices)
    elif (
        num_heads % (num_devices // 2) == 0
        and num_devices % 2 == 0
    ):
        mesh_shape = (2, num_devices // 2)
    else:
        raise ValueError(
            f"Cannot evenly distribute {num_heads} heads across {num_devices} devices"
        )
    return mesh_shape, ("batch", "model")

def load_shard_spec(self, model):
    shard_specs = {}
    # model is usually Wrapper(Qwen2_5_VLForConditionalGeneration)
    # We access the underlying HF model via .model
    root = model.model
    
    # --- Text Model Sharding ---
    # The text backbone is located at root.model.language_model
    text_layers = root.model.language_model.layers
    for layer in text_layers:
        # MLP
        shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
        shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
        shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

        # Attention
        shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.q_proj.bias] = ("model",)
        shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.k_proj.bias] = ("model",)
        shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.v_proj.bias] = ("model",)
        shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
    
    # LM Head
    shard_specs[root.lm_head.weight] = ("model", "batch")

    
    # --- Vision Model Sharding ---
    # The vision backbone is located at root.model.visual
    visual = root.model.visual
    
    # Patch Embedding
    # Conv3d(3, 1280, ...) -> Split output channels (dim 0)
    shard_specs[visual.patch_embed.proj.weight] = ("model", "batch")
    if visual.patch_embed.proj.bias is not None:
        shard_specs[visual.patch_embed.proj.bias] = ("model",)
    
    # Vision Blocks
    for block in visual.blocks:
        # MLP
        shard_specs[block.mlp.up_proj.weight] = ("model", "batch")
        shard_specs[block.mlp.gate_proj.weight] = ("model", "batch")
        shard_specs[block.mlp.down_proj.weight] = ("batch", "model")
        
        # Attention
        # qkv is a fused Linear(1280, 3840) -> Colwise split is safe
        shard_specs[block.attn.qkv.weight] = ("model", "batch")
        if block.attn.qkv.bias is not None:
            shard_specs[block.attn.qkv.bias] = ("model",)
        # proj is Linear(1280, 1280) -> Rowwise split
        shard_specs[block.attn.proj.weight] = ("batch", "model")

    # Vision Merger (Sequential MLP: Linear -> GELU -> Linear)
    # merger.mlp[0]: Linear(5120->5120) -> Colwise
    shard_specs[visual.merger.mlp[0].weight] = ("model", "batch")
    if visual.merger.mlp[0].bias is not None:
        shard_specs[visual.merger.mlp[0].bias] = ("model",)
    
    # merger.mlp[2]: Linear(5120->2048) -> Rowwise
    shard_specs[visual.merger.mlp[2].weight] = ("batch", "model")


    return shard_specs

'''
Qwen2_5_VLForConditionalGeneration(
  (model): Qwen2_5_VLModel(
    (visual): Qwen2_5_VisionTransformerPretrainedModel(
      (patch_embed): Qwen2_5_VisionPatchEmbed(
        (proj): Conv3d(3, 1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), bias=False)
      )
      (rotary_pos_emb): Qwen2_5_VisionRotaryEmbedding()
      (blocks): ModuleList(
        (0-31): 32 x Qwen2_5_VLVisionBlock(
          (norm1): Qwen2RMSNorm((1280,), eps=1e-06)
          (norm2): Qwen2RMSNorm((1280,), eps=1e-06)
          (attn): Qwen2_5_VLVisionAttention(
            (qkv): Linear(in_features=1280, out_features=3840, bias=True)
            (proj): Linear(in_features=1280, out_features=1280, bias=True)
          )
          (mlp): Qwen2_5_VLMLP(
            (gate_proj): Linear(in_features=1280, out_features=3420, bias=True)
            (up_proj): Linear(in_features=1280, out_features=3420, bias=True)
            (down_proj): Linear(in_features=3420, out_features=1280, bias=True)
            (act_fn): SiLU()
          )
        )
      )
      (merger): Qwen2_5_VLPatchMerger(
        (ln_q): Qwen2RMSNorm((1280,), eps=1e-06)
        (mlp): Sequential(
          (0): Linear(in_features=5120, out_features=5120, bias=True)
          (1): GELU(approximate='none')
          (2): Linear(in_features=5120, out_features=2048, bias=True)
        )
      )
    )
    (language_model): Qwen2_5_VLTextModel(
      (embed_tokens): Embedding(151936, 2048)
      (layers): ModuleList(
        (0-35): 36 x Qwen2_5_VLDecoderLayer(
          (self_attn): Qwen2_5_VLAttention(
            (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
            (k_proj): Linear(in_features=2048, out_features=256, bias=True)
            (v_proj): Linear(in_features=2048, out_features=256, bias=True)
            (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
            (rotary_emb): Qwen2_5_VLRotaryEmbedding()
          )
          (mlp): Qwen2MLP(
            (gate_proj): Linear(in_features=2048, out_features=11008, bias=False)
            (up_proj): Linear(in_features=2048, out_features=11008, bias=False)
            (down_proj): Linear(in_features=11008, out_features=2048, bias=False)
            (act_fn): SiLU()
          )
          (input_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
          (post_attention_layernorm): Qwen2RMSNorm((2048,), eps=1e-06)
        )
      )
      (norm): Qwen2RMSNorm((2048,), eps=1e-06)
      (rotary_emb): Qwen2_5_VLRotaryEmbedding()
    )
  )
  (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
)
'''