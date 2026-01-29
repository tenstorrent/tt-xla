# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import run_graph_test
from infra.evaluators.evaluation_config import ComparisonConfig, PccConfig
from infra.utilities import Framework
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.qwen_2_5_vl.pytorch.loader import ModelLoader


@pytest.mark.parametrize("tp_bool", [True, False])
def test_display_language_inputs(tp_bool):
    # Load model
    loader = ModelLoader()
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = (
        loader.load_inputs()
    )  # returns BatchFeature with input_ids, attention_mask, pixel_values, image_grid_thw, video_grid_thw, second_per_grid_ts

    # print(inputs)
    # for key, value in inputs.items():
    #    print(f"{key}: {value.shape}")

    input_ids = inputs["input_ids"]
    vl_model = model.model  # Qwen2_5_VLModel
    inputs_embeds = vl_model.get_input_embeddings()(input_ids)
    print(f"input_embeds: {inputs_embeds}")
    print(f"input_embeds.shape: {inputs_embeds.shape}")

    print(f"vl_model.config.hidden_size: {vl_model.config.hidden_size}")
    language_model = vl_model.language_model  # Qwen2_5_VLTextModel
    print(f"language_model.config.hidden_size: {language_model.config.hidden_size}")

    past_seen_tokens = 0
    cache_position = torch.arange(
        past_seen_tokens,
        past_seen_tokens + inputs_embeds.shape[1],
        device=inputs_embeds.device,
    )
    position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    print(f"position_ids: {position_ids}")
    print(f"position_ids.shape: {position_ids.shape}")

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = language_model.rotary_emb(hidden_states, position_ids)
    print(f"position_embeddings: {position_embeddings}")
    print(f"position_embeddings[0].shape: {position_embeddings[0].shape}")


@pytest.mark.parametrize("tp_bool", [True, False])
def test_qwen_2_5_vl_language_layer(tp_bool):
    """Test Qwen2.5-VL single language decoder layer with tensor parallel sharding."""
    # Load model
    loader = ModelLoader()
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = (
        loader.load_inputs()
    )  # returns BatchFeature with input_ids, attention_mask, pixel_values, image_grid_thw, video_grid_thw, second_per_grid_ts

    language_model = model.model.language_model  # Qwen2_5_VLTextModel
    config = language_model.config  # Qwen2_5_VLTextConfig

    # Extract language model layer
    decoder_layer = language_model.layers[0]  # Qwen2_5_VLDecoderLayer[0]

    print(f"\n{'='*60}")
    print(f"Testing Language Layer with tp={tp_bool}")
    print(f"{'='*60}\n")

    batch_size = 4
    seq_len = 1024
    hidden_size = config.hidden_size  # 2048
    head_dim = hidden_size // config.num_attention_heads

    # hidden states
    # input_ids from Qwen2_5_VLForConditionalGeneration passed to Qwen2_5_VLTextModel and made into inputs_embeds with
    # self.get_input_embeddings()(input_ids)
    # input_embeds then adds image/video embeddings if present
    # and then passes to Qwen2_5_VLTextModel
    # Qwen2_5_VLTextModel passes inputs_embeds to Qwen2_5_VLDecoderLayer[0] as hidden_states
    # should be (batch_size, seq_len, hidden_size)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # position embeddings
    # should be (3, batch_size, seq_len, head_dim)
    cos = torch.randn(3, batch_size, seq_len, head_dim, dtype=torch.bfloat16)
    sin = torch.randn(3, batch_size, seq_len, head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos, sin)

    # Forward signature:
    # (hidden_states, attention_mask, position_ids, past_key_values, output_attentions, use_cache, cache_position, position_embeddings)
    decoder_inputs = [
        hidden_states,
        None,
        None,
        None,
        False,
        False,
        None,
        position_embeddings,
    ]

    xr.set_device_type("TT")

    # Setup Mesh
    if tp_bool:
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (batch_size, num_devices // batch_size)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(layer, args, kwargs):
            shard_specs = {}
            # Attention
            shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
            # shard_specs[layer.self_attn.k_proj.weight] = (None, "batch")
            # shard_specs[layer.self_attn.v_proj.weight] = (None, "batch")
            shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

            # MLP
            shard_specs[layer.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[layer.mlp.down_proj.weight] = ("batch", "model")

            return shard_specs

    run_graph_test(
        decoder_layer,
        decoder_inputs,
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.98)),
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.parametrize("tp_bool", [True, False])
def test_qwen_2_5_vl_language_attention(tp_bool):
    """Test Qwen2.5-VL single language decoder layer attention with tensor parallel sharding."""
    # Load model
    loader = ModelLoader()
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = (
        loader.load_inputs()
    )  # returns BatchFeature with input_ids, attention_mask, pixel_values, image_grid_thw, video_grid_thw, second_per_grid_ts

    language_model = model.model.language_model  # Qwen2_5_VLTextModel
    config = language_model.config  # Qwen2_5_VLTextConfig

    # Extract language model layer
    decoder_layer = language_model.layers[0]  # Qwen2_5_VLDecoderLayer[0]

    attention = decoder_layer.self_attn  # Qwen2_5_VLAttention

    print(f"\n{'='*60}")
    print(f"Testing Language Layer attention with tp={tp_bool}")
    print(f"{'='*60}\n")

    batch_size = 1
    seq_len = 1024
    hidden_size = config.hidden_size  # 2048
    head_dim = hidden_size // config.num_attention_heads

    # hidden states
    # input_ids from Qwen2_5_VLForConditionalGeneration passed to Qwen2_5_VLTextModel and made into inputs_embeds with
    # self.get_input_embeddings()(input_ids)
    # input_embeds then adds image/video embeddings if present
    # and then passes to Qwen2_5_VLTextModel
    # Qwen2_5_VLTextModel passes inputs_embeds to Qwen2_5_VLDecoderLayer[0] as hidden_states
    # should be (batch_size, seq_len, hidden_size)
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # position embeddings
    # should be (3, batch_size, seq_len, head_dim)
    cos = torch.randn(3, batch_size, seq_len, head_dim, dtype=torch.bfloat16)
    sin = torch.randn(3, batch_size, seq_len, head_dim, dtype=torch.bfloat16)
    position_embeddings = (cos, sin)

    # Forward signature:
    # (hidden_states, attention_mask, position_ids, past_key_values, output_attentions, use_cache, cache_position, position_embeddings)
    attention_inputs = [
        hidden_states,
        None,
        None,
        None,
        False,
        False,
        None,
        position_embeddings,
    ]

    xr.set_device_type("TT")

    # Setup Mesh
    if tp_bool:
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (batch_size, num_devices // batch_size)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(attention, args, kwargs):
            shard_specs = {}
            # Attention
            shard_specs[attention.q_proj.weight] = ("model", "batch")
            # shard_specs[attention.k_proj.weight] = ("model", "batch")
            # shard_specs[attention.v_proj.weight] = ("model", "batch")
            shard_specs[attention.o_proj.weight] = ("batch", "model")

            return shard_specs

    run_graph_test(
        attention,
        attention_inputs,
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.98)),
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.parametrize("tp_bool", [True, False])
def test_qwen_2_5_vl_patch_embed(tp_bool):
    """Test Qwen2.5-VL patch embedding (Conv3d) with tensor parallel sharding."""
    loader = ModelLoader()
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = (
        loader.load_inputs()
    )  # returns BatchFeature with input_ids, attention_mask, pixel_values, image_grid_thw, video_grid_thw, second_per_grid_ts

    vl_model = model.model  # Qwen2_5_VLModel
    print(f"vl_model.config.hidden_size: {vl_model.config.hidden_size}")
    vision_model = vl_model.visual  # Qwen2_5_VisionTransformerPretrainedModel
    # takes input hidden_states and grid_thw
    hidden_states = inputs["pixel_values"]
    grid_thw = inputs["image_grid_thw"]
    print(f"hidden_states: {hidden_states.shape}")
    print(f"grid_thw: {grid_thw}")

    patch_embed = vision_model.patch_embed  # Qwen2_5_VisionPatchEmbed

    inputs = [hidden_states]

    xr.set_device_type("TT")

    # Setup Mesh
    mesh = None
    get_shard_spec = None
    if tp_bool:
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(layer_model, args, kwargs):
            shard_specs = {}
            # Shard Conv3d output channels (1280)
            # visual.patch_embed.proj is the Conv3d layer
            shard_specs[layer_model.proj.weight] = ("model", "batch")

            return shard_specs

    run_graph_test(
        patch_embed,
        inputs,
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.98)),
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


def generate_vision_block_inputs():
    # Load model
    loader = ModelLoader()
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = (
        loader.load_inputs()
    )  # returns BatchFeature with input_ids, attention_mask, pixel_values, image_grid_thw, video_grid_thw, second_per_grid_ts

    # print(inputs)
    for key, value in inputs.items():
        print(f"{key}: {value.shape}")

    vl_model = model.model  # Qwen2_5_VLModel
    print(f"vl_model.config.hidden_size: {vl_model.config.hidden_size}")
    vision_model = vl_model.visual  # Qwen2_5_VisionTransformerPretrainedModel
    # takes input hidden_states and grid_thw
    hidden_states = inputs["pixel_values"]
    grid_thw = inputs["image_grid_thw"]
    print(f"hidden_states: {hidden_states.shape}")
    print(f"grid_thw: {grid_thw}")

    patch_embed = vision_model.patch_embed  # Qwen2_5_VisionPatchEmbed

    hidden_states = patch_embed(hidden_states)
    print(f"hidden_states (patched): {hidden_states.shape}")

    rotary_pos_emb = vision_model.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = vision_model.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(
        seq_len // vision_model.spatial_merge_unit, vision_model.spatial_merge_unit, -1
    )
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    print(f"Final hidden_states.shape after some reshaping: {hidden_states.shape}")
    rotary_pos_emb = rotary_pos_emb.reshape(
        seq_len // vision_model.spatial_merge_unit, vision_model.spatial_merge_unit, -1
    )
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())
    # print(f"position_embeddings: {position_embeddings}")
    print(f"position_embeddings[0].shape: {position_embeddings[0].shape}")

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )

    import torch.nn.functional as F

    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
    print(f"cu_seqlens: {cu_seqlens}")
    print(f"cu_seqlens.shape: {cu_seqlens.shape}")

    return hidden_states, cu_seqlens, position_embeddings


@pytest.mark.parametrize("tp_bool", [True, False])
def test_qwen_2_5_vl_vision_block(tp_bool):
    """Test Qwen2.5-VL single vision transformer block with tensor parallel sharding."""
    # Load model
    loader = ModelLoader()
    model = loader.load_model(dtype_override=torch.bfloat16)

    vision_model = model.model.visual  # Qwen2_5_VisionTransformerPretrainedModel
    visual_block = model.model.visual.blocks[0]

    print(f"\n{'='*60}")
    print(f"Testing Vision Block with tp={tp_bool}")
    print(f"{'='*60}\n")

    hidden_states, cu_seqlens, position_embeddings = generate_vision_block_inputs()
    # Forward signature: (hidden_states, cu_seqlens, rotary_pos_emb=None, position_embeddings=None)
    inputs = [hidden_states, cu_seqlens, None, position_embeddings]

    xr.set_device_type("TT")

    if tp_bool:
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(visual_block, args, kwargs):
            shard_specs = {}

            # Attention
            shard_specs[visual_block.attn.qkv.weight] = ("model", "batch")
            shard_specs[visual_block.attn.proj.weight] = ("batch", "model")

            # MLP
            shard_specs[visual_block.mlp.gate_proj.weight] = ("model", "batch")
            shard_specs[visual_block.mlp.up_proj.weight] = ("model", "batch")
            shard_specs[visual_block.mlp.down_proj.weight] = ("batch", "model")

            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        visual_block,
        inputs,
        framework=Framework.TORCH,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.98)),
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


def load_config(self):
    """Load and return the configuration for the Qwen 2.5 VL model variant."""
    self.config = AutoConfig.from_pretrained(self._variant_config.pretrained_model_name)
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
    elif num_heads % (num_devices // 2) == 0 and num_devices % 2 == 0:
        mesh_shape = (2, num_devices // 2)
    else:
        raise ValueError(
            f"Cannot evenly distribute {num_heads} heads across {num_devices} devices"
        )

    # 4 by 2 override for testing
    mesh_shape = (4, num_devices // 4)
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
        # shard_specs[layer.self_attn.q_proj.bias] = ("model",)
        # shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
        # shard_specs[layer.self_attn.k_proj.bias] = ("model",)
        # shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
        # shard_specs[layer.self_attn.v_proj.bias] = ("model",)
        shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")

    # LM Head
    shard_specs[root.lm_head.weight] = ("model", "batch")

    # --- Vision Model Sharding ---
    # The vision backbone is located at root.model.visual
    visual = root.model.visual

    # Patch Embedding
    # Conv3d(3, 1280, ...) -> Split output channels (dim 0)
    # shard_specs[visual.patch_embed.proj.weight] = ("model", "batch", None, None, None)
    # if visual.patch_embed.proj.bias is not None:
    #    shard_specs[visual.patch_embed.proj.bias] = ("model",)

    # Vision Blocks
    """for block in visual.blocks:
        # MLP
        shard_specs[block.mlp.up_proj.weight] = ("model", "batch")
        shard_specs[block.mlp.gate_proj.weight] = ("model", "batch")
        shard_specs[block.mlp.down_proj.weight] = ("batch", "model")

        # Attention
        # qkv is a fused Linear(1280, 3840) -> Colwise split is safe
        shard_specs[block.attn.qkv.weight] = ("model", "batch")
        #if block.attn.qkv.bias is not None:
        #    shard_specs[block.attn.qkv.bias] = ("model",)
        # proj is Linear(1280, 1280) -> Rowwise split
        shard_specs[block.attn.proj.weight] = ("batch", "model")"""

    # Vision Merger (Sequential MLP: Linear -> GELU -> Linear)
    # merger.mlp[0]: Linear(5120->5120) -> Colwise
    # shard_specs[visual.merger.mlp[0].weight] = ("model", "batch")
    # if visual.merger.mlp[0].bias is not None:
    #    shard_specs[visual.merger.mlp[0].bias] = ("model",)

    # merger.mlp[2]: Linear(5120->2048) -> Rowwise
    # shard_specs[visual.merger.mlp[2].weight] = ("batch", "model")

    return shard_specs


"""
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
"""
