
from jax import grad, jit, vmap
import jax.numpy as jnp
import jax
import os
import sys
import jax._src.xla_bridge as xb
from jax.lax import GatherDimensionNumbers
import flax.linen as nn

def initialize(): 
    backend = "tt"
    path = os.path.join(os.path.dirname(__file__), "../../build/src/tt/pjrt_plugin_tt.so")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find tt_pjrt C API plugin at {path}")
    print("Loading tt_pjrt C API plugin", file=sys.stderr)
    xb.discover_pjrt_plugins()
    plugin = xb.register_plugin('tt', priority=500, library_path=path, options=None)
    print("Loaded", file=sys.stderr)
    jax.config.update("jax_platforms", "tt,cpu")

def jax_take():
    print("\n\n Before operand:\n\n")
    tensor = jnp.zeros((32000, 1024), dtype=jnp.float32)
    print("\n\nBefore start_indices:\n\n")
    indices = jnp.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]],dtype=jnp.int16)
    
    print("\n\nBefore take:\n\n")
    
    try:
        # Use jit to force compilation and IR generation
        @jax.jit
        def take_fn(tensor, indices):
            return jnp.take(tensor, indices, axis=0)
        
        print("\n\nBefore take:\n\n")
        gathered = take_fn(tensor, indices)
        print(gathered.shape)
    except Exception as e:
        print("Error:", e)

def jax_indexing():
    print("\n\n Before operand:\n\n")
    tensor = jnp.zeros((32000, 1024), dtype=jnp.float32)
    print("\n\nBefore start_indices:\n\n")
    indices = jnp.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]],dtype=jnp.int16)
    
    print("\n\nBefore indexing:\n\n")
    
    try:
        # Use jit to force compilation and IR generation
        @jax.jit
        def indexing_fn(tensor, indices):
            return tensor[indices[0]]
        
        print("\n\nBefore take:\n\n")
        gathered = indexing_fn(tensor, indices)
        print(gathered.shape)
    except Exception as e:
        print("Error:", e)

def jax_vmap():
    print("\n\n Before operand:\n\n")
    tensor = jnp.zeros((32000, 1024), dtype=jnp.float32)
    print("\n\nBefore start_indices:\n\n")
    indices = jnp.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                      16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]],dtype=jnp.int16)
    
    print("\n\nBefore vmap:\n\n")
    
    try:
        # Use jit to force compilation and IR generation
        @jax.jit
        def vmap_fn(index):
            return tensor[index]
        
        print("\n\nBefore vmap:\n\n")
        gathered = vmap(vmap_fn)(indices[0])
        print(gathered.shape)
    except Exception as e:
        print("Error:", e)

class EmbeddingModel(nn.Module):
    vocab_size: int
    embedding_dim: int

    @nn.compact
    def __call__(self, indices):
        embedding = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.embedding_dim,
            dtype=jnp.float32 
        )
        return embedding(indices)

def flax_embed():
    print("\n\nInitializing model:\n\n")
    
    # Model parameters
    vocab_size = 32000
    embedding_dim = 1024
    
    # Create and initialize the model
    model = EmbeddingModel(vocab_size=vocab_size, embedding_dim=embedding_dim)
    
    # Create sample indices
    indices = jnp.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]], 
                       dtype=jnp.int16)  # Changed to int32 as per Flax conventions
    
    print("\n\nBefore embedding lookup:\n\n")
    
    try:
        # Initialize parameters
        key = jax.random.PRNGKey(0)
        params = model.init(key, indices)
        
        # JIT the forward pass
        @jax.jit
        def embed_fn(params, indices):
            return model.apply(params, indices)
        
        print("\n\nPerforming embedding lookup:\n\n")
        embedded = embed_fn(params, indices)
        print(embedded.shape)
        
    except Exception as e:
        print("Error:", e)


if __name__ == "__main__":
    initialize()
    print("\n\nBefore valid_jax_gather_example\n\n")
    jax_take() # output sizes match with gather, fails during stablehlo

    # the following tests fail before shlo
    # flax_embed()
    # jax_indexing() # output shape does not match
    ## jax_vmap() # uses dynamic slice which fails
    