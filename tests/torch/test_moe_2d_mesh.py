import os

# MUST set XLA_EXPERIMENTAL BEFORE importing torch_xla
# Enables bounded dynamic shape support for nonzero, masked_select, etc.
os.environ["XLA_EXPERIMENTAL"] = "nonzero:masked_select:masked_scatter"

import transformers
print(transformers.__version__)
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh


# Number of chunks to split experts into (reduces peak memory)
# 128 experts / 1 chunk = 128 experts per chunk
# With 8-way expert sharding: 128 / 8 = 16 experts per device
MOE_NUM_CHUNKS = 1


def static_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Static MoE forward with expert sharding support.
    Uses batched einsum for parallel expert computation.
    Replaces Qwen3MoeSparseMoeBlock.forward for XLA compatibility.
    
    Supports chunked processing to reduce peak memory usage.
    """
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states_flat = hidden_states.view(-1, hidden_dim)
    num_tokens = batch_size * sequence_length
    
    # Router logits
    router_logits = self.gate(hidden_states_flat)
    
    # Softmax and top-k selection
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    
    if self.norm_topk_prob:
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    
    # Create full routing weight matrix [num_tokens, num_experts]
    full_routing_weights = torch.zeros(
        num_tokens, self.num_experts, 
        dtype=hidden_states.dtype, device=hidden_states.device
    )
    full_routing_weights.scatter_(1, selected_experts, routing_weights)
    
    # Lazy initialization of chunked batched weights if not already done
    # Note: If apply_expert_sharding was called, _chunked_weights is already set with XLA tensors
    if not hasattr(self, '_chunked_weights') or self._chunked_weights is None:
        init_chunked_weights(self, MOE_NUM_CHUNKS)
    
    # Chunked expert computation
    # Process experts in chunks to reduce peak memory
    final_hidden_states = torch.zeros(
        num_tokens, hidden_dim,
        dtype=hidden_states.dtype, device=hidden_states.device
    )
    
    experts_per_chunk = self.num_experts // MOE_NUM_CHUNKS
    
    for chunk_idx in range(MOE_NUM_CHUNKS):
        start_expert = chunk_idx * experts_per_chunk
        end_expert = start_expert + experts_per_chunk
        
        # Get weights for this chunk
        w1_chunk = self._chunked_weights['w1'][chunk_idx]
        w2_chunk = self._chunked_weights['w2'][chunk_idx]
        w3_chunk = self._chunked_weights['w3'][chunk_idx]
        
        # Batched expert computation using einsum
        # gate_proj: x @ w1.T -> [num_tokens, intermediate_size] for each expert in chunk
        gate_out = torch.einsum('th,eih->eti', hidden_states_flat, w1_chunk)
        up_out = torch.einsum('th,eih->eti', hidden_states_flat, w3_chunk)
        
        # SiLU activation and element-wise multiply
        activated = F.silu(gate_out) * up_out  # [chunk_experts, num_tokens, intermediate_size]
        
        # down_proj: activated @ w2.T -> [num_tokens, hidden_size] for each expert
        expert_outputs = torch.einsum('eti,ehi->eth', activated, w2_chunk)
        # expert_outputs: [chunk_experts, num_tokens, hidden_size]
        
        # Apply routing weights for this chunk of experts
        chunk_routing_weights = full_routing_weights[:, start_expert:end_expert]  # [num_tokens, chunk_experts]
        weights = chunk_routing_weights.t().unsqueeze(-1)  # [chunk_experts, num_tokens, 1]
        
        # Weighted sum for this chunk
        weighted_outputs = expert_outputs * weights
        chunk_result = weighted_outputs.sum(dim=0)  # [num_tokens, hidden_size]
        
        # Accumulate results
        final_hidden_states = final_hidden_states + chunk_result
    
    final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
    return final_hidden_states, router_logits


def init_chunked_weights(moe_block, num_chunks, device=None):
    """
    Initialize chunked batched weights for memory-efficient MoE.
    """
    experts = moe_block.experts
    num_experts = len(experts)
    experts_per_chunk = num_experts // num_chunks
    
    print(f"  Initializing chunked weights: {num_experts} experts / {num_chunks} chunks = {experts_per_chunk} per chunk")
    
    chunked_weights = {
        'w1': [],
        'w2': [],
        'w3': [],
    }
    
    for chunk_idx in range(num_chunks):
        start = chunk_idx * experts_per_chunk
        end = start + experts_per_chunk
        
        # Stack weights for this chunk (detach and move to CPU first for clean copy)
        w1_chunk = torch.stack([experts[i].gate_proj.weight.detach() for i in range(start, end)], dim=0)
        w2_chunk = torch.stack([experts[i].down_proj.weight.detach() for i in range(start, end)], dim=0)
        w3_chunk = torch.stack([experts[i].up_proj.weight.detach() for i in range(start, end)], dim=0)
        
        # Move to target device if specified
        if device is not None:
            w1_chunk = w1_chunk.to(device)
            w2_chunk = w2_chunk.to(device)
            w3_chunk = w3_chunk.to(device)
        
        chunked_weights['w1'].append(w1_chunk)
        chunked_weights['w2'].append(w2_chunk)
        chunked_weights['w3'].append(w3_chunk)
    
    moe_block._chunked_weights = chunked_weights
    print(f"  Chunk weight shapes: w1={chunked_weights['w1'][0].shape}, device={chunked_weights['w1'][0].device}")


def init_all_expert_weights(model, num_chunks=MOE_NUM_CHUNKS, device=None):
    """
    Initialize chunked weights for all MoE blocks in the model.
    """
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    
    print(f"\n=== Initializing Expert Weights ({num_chunks} chunks) ===")
    for name, module in model.named_modules():
        if isinstance(module, Qwen3MoeSparseMoeBlock):
            print(f"  {name}")
            init_chunked_weights(module, num_chunks, device=device)


def apply_expert_sharding_2d(model, mesh, num_chunks=MOE_NUM_CHUNKS):
    """
    Apply expert sharding on "expert" axis (8-way).
    
    2D Mesh: ("hidden", "expert") = (1, 8)
    
    Expert weights shard on expert dim only.
    
    w1 (gate): [experts, intermediate, hidden] → ("expert", None, None)
    w2 (down): [experts, hidden, intermediate] → ("expert", None, None)
    w3 (up):   [experts, intermediate, hidden] → ("expert", None, None)
    
    128 experts / 8 = 16 experts per device
    """
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    
    for name, module in model.named_modules():
        if isinstance(module, Qwen3MoeSparseMoeBlock):
            print(f"Applying expert sharding to {name}")
            
            for chunk_idx in range(num_chunks):
                w1_chunk = module._chunked_weights['w1'][chunk_idx]
                w2_chunk = module._chunked_weights['w2'][chunk_idx]
                w3_chunk = module._chunked_weights['w3'][chunk_idx]
                
                # Expert parallelism on "expert" axis (8-way)
                # 128/8 = 16 experts per device
                xs.mark_sharding(w1_chunk, mesh, ("expert", None, None))  # [E/8, I, H]
                xs.mark_sharding(w2_chunk, mesh, ("expert", None, None))  # [E/8, H, I]
                xs.mark_sharding(w3_chunk, mesh, ("expert", None, None))  # [E/8, I, H]
            
            print(f"  Sharded: 128/8 = 16 experts per device")


def apply_model_sharding_2d(model, mesh):
    """
    Apply tensor parallelism with 2D mesh.
    
    2D Mesh: ("hidden", "expert") = (1, 8)
    
    Use "expert" axis (8-way) for model weight sharding.
    Column parallel for Q/K/V → num_heads sharded, head_dim replicated!
    """
    print("\n=== Applying Model Weight Sharding (2D Mesh) ===")
    print("  Using 'expert' axis (8-way) for model parallelism")
    
    # Shard embedding layer: [vocab_size, hidden_size] -> shard hidden on "expert" (8-way)
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed = model.model.embed_tokens
        xs.mark_sharding(embed.weight, mesh, (None, "expert"))
        print(f"  Sharded embed_tokens: {embed.weight.shape} → (None, 'expert')")
    
    # Shard LM head: [vocab_size, hidden_size] -> shard hidden on "expert" (8-way)
    if hasattr(model, 'lm_head'):
        xs.mark_sharding(model.lm_head.weight, mesh, (None, "expert"))
        print(f"  Sharded lm_head: {model.lm_head.weight.shape} → (None, 'expert')")
    
    # Shard other weights
    for name, module in model.named_modules():
        # RMSNorm / LayerNorm weight sharding on "expert" axis (8-way)
        if ('norm' in name or 'layernorm' in name) and hasattr(module, 'weight'):
            if module.weight is not None and len(module.weight.shape) == 1:
                xs.mark_sharding(module.weight, mesh, ("expert",))
                print(f"  Sharded norm weight: {name} {module.weight.shape} → ('expert',)")
        
        # Attention layers - COLUMN PARALLEL on "expert" (8-way)
        # This shards num_heads, NOT head_dim → RoPE doesn't need CCL!
        if 'self_attn' in name and hasattr(module, 'q_proj'):
            # Q/K/V: COLUMN parallel (shard output dim = num_heads direction)
            if hasattr(module, 'q_proj') and module.q_proj.weight is not None:
                xs.mark_sharding(module.q_proj.weight, mesh, (None, "expert"))
            if hasattr(module, 'k_proj') and module.k_proj.weight is not None:
                xs.mark_sharding(module.k_proj.weight, mesh, (None, "expert"))
            if hasattr(module, 'v_proj') and module.v_proj.weight is not None:
                xs.mark_sharding(module.v_proj.weight, mesh, (None, "expert"))
            
            # O: ROW parallel (shard input dim)
            if hasattr(module, 'o_proj') and module.o_proj.weight is not None:
                xs.mark_sharding(module.o_proj.weight, mesh, ("expert", None))
            print(f"  Sharded attention (col-parallel): {name}")
        
        # Shared expert in MoE layers - same pattern
        if 'shared_expert' in name and hasattr(module, 'gate_proj'):
            # gate/up: column parallel (shard output = intermediate)
            if hasattr(module, 'gate_proj') and module.gate_proj.weight is not None:
                xs.mark_sharding(module.gate_proj.weight, mesh, (None, "expert"))
            if hasattr(module, 'up_proj') and module.up_proj.weight is not None:
                xs.mark_sharding(module.up_proj.weight, mesh, (None, "expert"))
            # down: row parallel (shard input = intermediate)
            if hasattr(module, 'down_proj') and module.down_proj.weight is not None:
                xs.mark_sharding(module.down_proj.weight, mesh, ("expert", None))
            print(f"  Sharded shared_expert (col-parallel): {name}")


def patch_qwen_moe_to_static():
    """Monkey-patch Qwen3 MoE to use static forward."""
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    Qwen3MoeSparseMoeBlock.forward = static_moe_forward
    print("Patched Qwen3MoeSparseMoeBlock.forward with static version")


def test_moe_model_2d_mesh():
    """
    Test MoE model with 2D mesh sharding.
    
    Mesh: ("hidden", "expert") = (1, 8) = 8 devices total
    
    - Hidden parallelism: 1-way (no sharding on hidden axis)
    - Expert parallelism: 8-way (128 experts / 8 = 16 experts per device)
    """
    print("Starting test_moe_model_2d_mesh")
    os.system('tt-smi -r')
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    os.environ["TTMLIR_RUNTIME_LOGGER_LEVEL"] = "Debug"
    xr.use_spmd()
    
    xr.set_device_type("TT")
    
    model_id = "Qwen/Qwen3-30B-A3B"
    config = AutoConfig.from_pretrained(
            model_id,
            # num_hidden_layers=1,  # Reduce for faster testing
            use_cache=False,
            attn_implementation="eager",
        )
    
    # Print attention config for debugging
    print(f"Config: num_attention_heads={config.num_attention_heads}, "
          f"num_key_value_heads={config.num_key_value_heads}, "
          f"hidden_size={config.hidden_size}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model in fp32 for CPU accuracy
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=torch.float32,
    )
    model.eval()

    SEQ_LEN = 64
    
    inputs = tokenizer(
        "Hello, my dog is cute",
        return_tensors="pt",
        padding="max_length",
        max_length=SEQ_LEN,
        truncation=True,
    )

    # Get actual input length (excluding padding)
    actual_len = inputs["attention_mask"].sum().item()
    print(f'Actual input length: {actual_len} tokens')
    print(f'Input text: {tokenizer.decode(inputs["input_ids"][0][:actual_len])}')

    # ============================================
    # 1. Original HuggingFace Model - CPU Result
    # ============================================
    print("\n=== Original HuggingFace Model (CPU) ===")
    with torch.no_grad():
        original_cpu_out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=None,
            use_cache=False,
        )
    original_cpu_logits = original_cpu_out.logits
    original_next_token_id = original_cpu_logits[0, actual_len-1].argmax().item()
    original_next_token = tokenizer.decode([original_next_token_id])
    print(f'Original CPU next token prediction: "{original_next_token}"')
    print(f'Original CPU logits[0, {actual_len-1}, :5]: {original_cpu_logits[0, actual_len-1, :5]}')

    # ============================================
    # 2. Apply Static MoE Patch
    # ============================================
    patch_qwen_moe_to_static()

    # ============================================
    # 3. Patched Model (Static MoE) - CPU Result
    # ============================================
    print("\n=== Patched Model (Static MoE) CPU ===")
    with torch.no_grad():
        patched_cpu_out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            position_ids=None,
            use_cache=False,
        )
    patched_cpu_logits = patched_cpu_out.logits
    patched_next_token_id = patched_cpu_logits[0, actual_len-1].argmax().item()
    patched_next_token = tokenizer.decode([patched_next_token_id])
    print(f'Patched CPU next token prediction: "{patched_next_token}"')
    print(f'Patched CPU logits[0, {actual_len-1}, :5]: {patched_cpu_logits[0, actual_len-1, :5]}')

    # Verify patch correctness
    if original_next_token_id == patched_next_token_id:
        print("✅ Patch verification: Original == Patched (next token matches)")
    else:
        print("⚠️ Patch verification: Original != Patched (next token differs!)")
    
    # Store for later comparison
    cpu_out_logits = patched_cpu_logits
    next_token = patched_next_token

    # ============================================
    # 4. XLA Setup
    # ============================================
    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))

    # 2D Mesh for expert parallelism
    # hidden: 1-way (no sharding)
    # expert: 8-way (128 experts / 8 = 16 per device)
    mesh_shape = (1, 8)
    mesh = Mesh(device_ids, mesh_shape, ("hidden", "expert"))
    
    print(f"\n=== 2D Mesh Configuration ===")
    print(f"  Shape: {mesh_shape}")
    print(f"  Axes: ('hidden', 'expert')")
    print(f"  Hidden parallelism: 1-way (no sharding)")
    print(f"  Expert parallelism: 8-way (128 / 8 = 16 experts per device)")

    xr.use_spmd()
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    
    # Initialize chunked expert weights on CPU FIRST (still fp32)
    print("\n=== Pre-initializing Expert Weights on CPU ===")
    init_all_expert_weights(model, num_chunks=MOE_NUM_CHUNKS, device=None)
    
    # Convert to bfloat16 and move to XLA
    print("\n=== Converting to bfloat16 and moving to XLA ===")
    model = model.to(torch.bfloat16).to("xla")
    
    # Convert chunked weights to bfloat16 and move to XLA
    print("\n=== Converting Chunked Weights to bfloat16 and moving to XLA ===")
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    for name, module in model.named_modules():
        if isinstance(module, Qwen3MoeSparseMoeBlock) and hasattr(module, '_chunked_weights'):
            for key in ['w1', 'w2', 'w3']:
                module._chunked_weights[key] = [
                    chunk.to(torch.bfloat16).to("xla") for chunk in module._chunked_weights[key]
                ]
            print(f"  Moved {name} chunked weights to XLA (bfloat16)")
    
    # Apply 2D sharding
    apply_model_sharding_2d(model, mesh)
    
    print("\n=== Applying 2D Expert Sharding ===")
    apply_expert_sharding_2d(model, mesh)
    
    input_ids = inputs["input_ids"].to("xla")
    attention_mask = inputs["attention_mask"].to("xla")
    position_ids = torch.arange(SEQ_LEN, dtype=torch.long).unsqueeze(0).to("xla")
    
    print("\nRunning forward pass with 2D mesh sharding...")
    
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    )
    
    print(f'output logits shape: {output.logits.shape}')
    logits_xla = output.logits.cpu()
    
    # ============================================
    # 5. Final Comparison: Original vs Patched vs XLA
    # ============================================
    print(f'\n=== Logits Comparison (at position {actual_len-1}) ===')
    print(f'Original CPU logits: {original_cpu_logits[0, actual_len-1, :5]}')
    print(f'Patched CPU logits:  {cpu_out_logits[0, actual_len-1, :5]}')
    print(f'XLA logits:          {logits_xla[0, actual_len-1, :5]}')
    
    # Next token prediction comparison
    xla_next_token_id = logits_xla[0, actual_len-1].argmax().item()
    xla_next_token = tokenizer.decode([xla_next_token_id])
    
    print(f'\n=== Next Token Predictions ===')
    print(f'Original CPU: "{original_next_token}" (id={original_next_token_id})')
    print(f'Patched CPU:  "{patched_next_token}" (id={patched_next_token_id})')
    print(f'XLA:          "{xla_next_token}" (id={xla_next_token_id})')
    
    # Summary
    print(f'\n=== Summary ===')
    if original_next_token_id == patched_next_token_id == xla_next_token_id:
        print("✅ All three match! (Original == Patched == XLA)")
    elif original_next_token_id == patched_next_token_id:
        print("⚠️ Original == Patched, but XLA differs")
    elif patched_next_token_id == xla_next_token_id:
        print("✅ Patched == XLA (XLA correctly reproduces patched model)")
    else:
        print("❌ All three differ!")


if __name__ == "__main__":
    test_moe_model_2d_mesh()

