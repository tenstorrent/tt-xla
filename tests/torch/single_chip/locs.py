import os
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr


# Set up XLA runtime for TT backend
xr.set_device_type("TT")

cache_dir = f"{os.getcwd()}/cache_pytorch_codegen"
xr.initialize_cache(cache_dir)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization as used in Llama"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class SwiGLU(nn.Module):
    """SwiGLU activation function as used in Llama's feed-forward network"""
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(nn.functional.silu(gate) * up)


class LlamaAttention(nn.Module):
    """Multi-head attention without RoPE to avoid graph breaks"""
    def __init__(self, hidden_size, num_heads, max_position_embeddings=2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        if self.head_dim * num_heads != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_heads")

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.size()

        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores (without RoPE)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / (self.head_dim ** 0.5)
        
        # Apply causal mask
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        attn_weights = attn_weights.masked_fill(causal_mask.to(attn_weights.device), float('-inf'))
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class LlamaDecoderLayer(nn.Module):
    """A single transformer decoder layer as used in Llama"""
    def __init__(self, hidden_size, num_heads, intermediate_size, max_position_embeddings=2048):
        super().__init__()
        self.hidden_size = hidden_size
        self.self_attn = LlamaAttention(hidden_size, num_heads, max_position_embeddings)
        self.mlp = SwiGLU(hidden_size, intermediate_size)
        self.input_layernorm = RMSNorm(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)

    def forward(self, hidden_states):
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        # Feed-forward with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class LlamaModel(nn.Module):
    """Complete Llama-like model architecture"""
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=512,
        intermediate_size=1376,  # ~2.7 * hidden_size
        num_hidden_layers=8,
        num_attention_heads=8,
        max_position_embeddings=2048,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Transformer layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(
                hidden_size=hidden_size,
                num_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                max_position_embeddings=max_position_embeddings,
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = RMSNorm(hidden_size)
        
        # Output projection (language modeling head)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, inputs_embeds):
        # Start with precomputed embeddings
        hidden_states = inputs_embeds

        # Pass through transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Return mean of logits for simplicity (in practice you'd use specific loss)
        return torch.mean(logits)


# Create model and input on TT device
device = xm.xla_device()

# Create a small but complete Llama model
model = LlamaModel(
    vocab_size=100,     # Small vocab for faster compilation
    hidden_size=12,     # Reduced size for testing
    intermediate_size=64, # ~2.7 * hidden_size
    num_hidden_layers=2, # Fewer layers for testing
    num_attention_heads=4,
    max_position_embeddings=128,
).to(device)

model = torch.compile(model, backend="tt")

# Create precomputed embedding vectors (batch_size=2, seq_len=32, hidden_size=256)
x = torch.randn(2, 32, 12).to(device)

# Execute the model to trigger compilation and caching
output = model(x)
output.to("cpu")

print(output[0])