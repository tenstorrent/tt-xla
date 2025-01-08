# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from collections import defaultdict
from jax import numpy as jnp


def ddict():
    return defaultdict(ddict)


def defaultdict_to_dict(d):
    """Recursively convert defaultdicts to dicts."""
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    elif isinstance(d, dict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d


# TODO(stefan): Similar logic might be needed in other places later
#               generalize and move to infra once it's needed
def build_pytee_from_npy(npfile):
    """Convert a file from numpy.load with keys of form a/b/c... into a pytree"""
    weights = ddict()
    for name, w in npfile.items():
        keys = list(name.split("/"))
        subdict = weights
        for key in keys[:-1]:
            subdict = subdict[key]
        subdict[keys[-1]] = jnp.array(w)
    return {"params": defaultdict_to_dict(weights)}
