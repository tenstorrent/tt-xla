# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Debug hook (``TT_TORCH_DEBUG_BRIDGE_INPUTS=1``): log how torch-xla's dynamo
bridge feeds each compiled graph. Off by default, nothing is patched.

Wraps ``GraphInputMatcher.__init__`` / ``__call__`` and logs, per graph, the
HLO inputs classified as trace-time constants, a WARNING for any argument left
with no HLO input, and per call the (shape, sum) fingerprints of the caller's
arguments vs. the cached constants actually fed. A stale input shows as a
caller argument whose fingerprint changes between calls while the cached
constant's does not. See ``CompiledModule._isolate_user_inputs``.
"""

import os

import torch
from ttxla_tools.logging import logger

ENV_VAR = "TT_TORCH_DEBUG_BRIDGE_INPUTS"
# The 295-input Llama body graph is too noisy to log arg-by-arg; the small
# graphs (lm_head: 2 args, stack: 32 args) are the interesting ones.
MAX_ARGS_TO_LOG = 40

_installed = False


def _fingerprint(t):
    """(shape, float32 sum) of a tensor. Forces a device->host read."""
    try:
        if not isinstance(t, torch.Tensor):
            return repr(t)
        return (
            tuple(t.shape),
            round(float(t.detach().to(torch.float32).sum().item()), 3),
        )
    except Exception as e:  # instrumentation must never break execution
        return (tuple(getattr(t, "shape", ())), f"<{type(e).__name__}>")


def install_bridge_input_debug() -> bool:
    """Patch torch-xla's GraphInputMatcher if the env var is set. Idempotent."""
    global _installed
    if _installed or os.environ.get(ENV_VAR, "0") != "1":
        return _installed

    from torch_xla._dynamo import dynamo_bridge

    Matcher = dynamo_bridge.GraphInputMatcher
    orig_init = Matcher.__init__
    orig_call = Matcher.__call__

    def patched_init(self, tensor_id_to_arg_idx, graph_input_tensor_ids, *args, **kw):
        orig_init(self, tensor_id_to_arg_idx, graph_input_tensor_ids, *args, **kw)
        self._tt_calls = 0
        self._tt_tag = (
            f"graph(n_args={len(tensor_id_to_arg_idx)}, "
            f"n_hlo_inputs={len(graph_input_tensor_ids)})"
        )
        matched = {i for i in self.arg_idxs if i is not None}
        unmatched_args = sorted(set(tensor_id_to_arg_idx.values()) - matched)
        const_inputs = [
            (tid, _fingerprint(val))
            for tid, idx, val in zip(
                self.graph_input_tensor_ids, self.arg_idxs, self.graph_input_xla_values
            )
            if idx is None
        ]
        logger.info(
            f"[BRIDGE] {self._tt_tag} compiled. arg tensor ids={sorted(tensor_id_to_arg_idx)}"
        )
        logger.info(
            f"[BRIDGE] {self._tt_tag} HLO inputs classified as CONSTANT "
            f"(captured at trace, reused every call) (tensor_id, (shape, sum)): {const_inputs}"
        )
        if unmatched_args:
            logger.warning(
                f"[BRIDGE] {self._tt_tag} ARGUMENTS WITH NO HLO INPUT -> the graph will "
                f"read a cached constant instead of the caller's value: "
                f"arg_idx={unmatched_args} tensor_id="
                f"{[t for t, i in tensor_id_to_arg_idx.items() if i in unmatched_args]}"
            )

    def patched_call(self, args):
        self._tt_calls += 1
        if len(args) <= MAX_ARGS_TO_LOG:
            logger.info(
                f"[BRIDGE] {self._tt_tag} call {self._tt_calls}: caller args "
                f"(idx, (shape, sum)) = {[(i, _fingerprint(a)) for i, a in enumerate(args)]}"
            )
        cached = [
            (hlo_idx, tid, _fingerprint(val))
            for hlo_idx, (tid, val, idx) in enumerate(
                zip(self.graph_input_tensor_ids, self.graph_input_xla_values, self.arg_idxs)
            )
            if idx is None and val is not None
        ]
        logger.info(
            f"[BRIDGE] {self._tt_tag} call {self._tt_calls}: HLO inputs fed from CACHED "
            f"CONSTANTS (hlo_idx, tensor_id, (shape, sum)) = {cached}"
        )
        return orig_call(self, args)

    Matcher.__init__ = patched_init
    Matcher.__call__ = patched_call
    _installed = True
    logger.warning(f"[BRIDGE] {ENV_VAR}=1: torch-xla GraphInputMatcher instrumented")
    return True
