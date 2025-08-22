import jax
import jax.numpy as jnp
from jax.tree_util import tree_map_with_path, DictKey, SequenceKey

import optax
import quax

from .constants import LORA_FREEZE, LORA_FULL
from .transform import LoraWeight


def init_lora(param_tree, spec, rng, stddev=0.01, dtype=jnp.float32, alpha=1., is_leaf=None):
    def iter_keys(key):
        while True:
            key, out_key = jax.random.split(key)
            yield out_key

    key_it = iter_keys(rng)

    def get_param(path, param, spec_val):
        if spec_val in (LORA_FREEZE, LORA_FULL):
            return param

        if len(param.shape) == 1:
            raise ValueError(f'Vectors must either be frozen or fully tuned, but got spec value {spec} for param with path {path}')

        if len(param.shape) == 2:
            b_dim, a_dim = param.shape

            b = jnp.zeros((b_dim, spec_val), dtype=dtype)
            a = jax.random.normal(next(key_it), (spec_val, a_dim), dtype=dtype) * stddev
            return LoraWeight(w=param, a=a, b=b, alpha=alpha)

        # conv case
        *window_shape, in_channels, out_channels = param.shape

        a = jnp.zeros((
            *(1 for _ in range(len(window_shape))),
            spec_val,
            out_channels
        ), dtype=param.dtype)
        b = jax.random.normal(rng, (*window_shape, in_channels, spec_val), dtype=param.dtype) * stddev
        return LoraWeight(param, a, b, alpha=alpha)

    return jax.tree_util.tree_map_with_path(get_param, param_tree, spec, is_leaf=is_leaf)

def simple_spec(params, decision_fn=None, tune_vectors=False, is_leaf=None):
    """
    Create a simple lora spec for a pytree
    Args:
        params: pytree of parameters
        tune_vectors: If true, will flag all arrays with less than 2 dimensions for tuning
        decision_fn: A function which maps a Jax KeyPath and a parameter to a spec value
    """
    if decision_fn is None:
        def decision_fn(*args):
            return LORA_FREEZE

    def full_fn(path, arr):
        if len(arr.shape) < 2:
            return LORA_FULL if tune_vectors else LORA_FREEZE

        value = decision_fn(path, arr)
        return value

    return tree_map_with_path(full_fn, params, is_leaf=is_leaf)

def merge_params(lora_params, destructive=True, use_scaling=True):
    """
    Re-merge LoRA parameters.
    Arguments:
        destructive: If True, the buffers in frozen_params may be freed to save memory.
        use_scaling: Whether to multiply LoRA params by alpha/r
    """
    if not use_scaling:
        raise ValueError('Scaling is now always enabled to match the original LoRA implementation.')

    def map_fn(param):
        if isinstance(param, LoraWeight):
            result = param.materialise()
            # Skip destructive deletion for now to avoid JAX API issues
            return result
        return param

    # Use tree_map with is_leaf to handle LoraWeight objects properly
    return jax.tree.map(map_fn, lora_params, is_leaf=lambda x: isinstance(x, LoraWeight))

def split_lora_params(params, spec):
    """
    Map params to a pytree in which all `LoraWeight.w` values and all params marked with
    LORA_FREEZE are replaced with None. This is useful for checkpointing just
    the trainable params.
    """
    def node_mapper(node, spec_val):
        if not isinstance(node, LoraWeight):
            return node if spec_val != LORA_FREEZE else None
        # Create a new LoraWeight with w=None to save memory during checkpointing
        return LoraWeight(w=None, a=node.a, b=node.b, alpha=node.alpha)

    return jax.tree.map(node_mapper, params, spec)

def wrap_optimizer(optimizer : optax.GradientTransformation, spec, scalar_frozen_grads=False):
    """
    Wrap the optimizer to freeze the 'w' component of LoraWeight objects.
    Only the 'a' and 'b' matrices should be trainable in LoRA.
    """
    def freeze_lora_weights(updates):
        def freeze_if_lora(update):
            if isinstance(update, LoraWeight):
                # For LoraWeight: freeze 'w' and 'alpha', allow 'a' and 'b' to update
                return LoraWeight(
                    w=jnp.zeros_like(update.w),  # Zero gradient for frozen w
                    a=update.a,                  # Keep gradient for a
                    b=update.b,                  # Keep gradient for b
                    alpha=0.0                    # Freeze alpha
                )
            else:
                # Keep updates for regular parameters
                return update
        
        return jax.tree.map(freeze_if_lora, updates, is_leaf=lambda x: isinstance(x, LoraWeight))
    
    # Chain the freezing with the original optimizer
    return optax.chain(
        optax.GradientTransformation(
            init=lambda params: {},
            update=lambda updates, state, params: (freeze_lora_weights(updates), state)
        ),
        optimizer
    )