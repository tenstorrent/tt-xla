import re
from flax.core.frozen_dict import freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from jax.sharding import PartitionSpec as P
import jax
from jax.sharding import NamedSharding
from jaxtyping import PyTree

# Sentinels
_unmatched = object()

# For specifying empty leaf dict `{}`
empty_dict = object()


def _match(qs, ks):
    """Return True if regexes in qs match any window of strings in tuple ks."""
    # compile regexes and force complete match
    qts = tuple(map(lambda x: re.compile(x + "$"), qs))
    for i in range(len(ks) - len(qs) + 1):
        matches = [x.match(y) for x, y in zip(qts, ks[i:])]
        if matches and all(matches):
            return True
    return False


def _replacement_rules(rules):
    def replace(key, val):
        for rule, replacement in rules:
            if _match(rule, key):
                return replacement
        return val

    return replace

def get_partition_spec(in_dict, rules):
    replace = _replacement_rules(rules)
    initd = {k: _unmatched for k in flatten_dict(in_dict)}
    result = {k: replace(k, v) for k, v in initd.items()}
    assert _unmatched not in result.values(), "Incomplete partition spec."
    return freeze(unflatten_dict(result))

def _get_partition_rules_llama(fsdp: bool=False):
    if fsdp:
        return [
            # embeddings
            (("transformer", "wte", "embedding"), P("mp", "dp")), 
            # atention
            (("attention", "(wq|wk|wv)", "kernel"), P("dp", "mp")), 
            (("attention", "wo", "kernel"), P("mp", "dp")), 
            # mlp
            (("feed_forward", "w1", "kernel"), P("dp", "mp")), 
            (("feed_forward", "w2", "kernel"), P("mp", "dp")), 
            (("feed_forward", "w3", "kernel"), P("dp", "mp")), 
            # layer norms
            (("attention_norm", "kernel"), P(None)),
            (("ffn_norm", "kernel"), P(None)),
            # output head
            (("transformer", "ln_f", "kernel"), P(None)), 
            (("lm_head", "kernel"), P("dp", "mp")), 
        ]
    return [
        # embeddings
        (("transformer", "wte", "embedding"), P("mp", None)), 
        # atention
        (("attention", "(wq|wk|wv)", "kernel"), P(None, "mp")), 
        (("attention", "wo", "kernel"), P("mp", None)), 
        # mlp
        (("feed_forward", "w1", "kernel"), P(None, "mp")), 
        (("feed_forward", "w2", "kernel"), P("mp", None)), 
        (("feed_forward", "w3", "kernel"), P(None, "mp")), 
        # layer norms
        (("attention_norm", "kernel"), P(None)),
        (("ffn_norm", "kernel"), P(None)),
        # output head
        (("transformer", "ln_f", "kernel"), P(None)), 
        (("lm_head", "kernel"), P(None, "mp")), 
    ]

def get_llama_param_partition_spec(params: PyTree, fsdp: bool=False) -> PyTree:
    return get_partition_spec(params, _get_partition_rules_llama(fsdp=fsdp))

def global_mesh_defined():
    """Checks if global xmap/pjit mesh resource environment is defined."""
    maps_env = jax.experimental.maps.thread_resources.env
    return maps_env.physical_mesh.devices.shape != ()  # pylint: disable=g-explicit-bool-comparison

def with_sharding_constraint(x, axis_resources):
    """Wrapper for pjit with_sharding_constraint, no-op on cpu or outside pjit."""
    if jax.devices()[0].platform == 'cpu' or not global_mesh_defined():
        return x
    else:
        return jax.lax.with_sharding_constraint(x, axis_resources)

def with_named_sharding_constraint(x, mesh, partition_spec):
    if mesh is not None:
        return with_sharding_constraint(x, NamedSharding(mesh, partition_spec))
    return x
