# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC
"""TT-compatible, pure-PyTorch Gated Delta Net ops.

Each op mirrors the signature of its counterpart in
``vllm.model_executor.layers.fla.ops`` / the standalone ``flash-linear-attention``
package, so a golden produced by FLA on a GPU box can drive the TT op directly
with no adapter. The implementations are plain PyTorch (no Triton, no
``torch.ops.vllm.*``, no global forward-context reads) so they lower cleanly
through ``torch.compile(backend="tt")`` -> tt-mlir.
"""

from .conv1d import tt_causal_conv1d_fn, tt_causal_conv1d_update
from .chunk import tt_chunk_gated_delta_rule
from .gating import tt_fused_gdn_gating
from .l2norm import tt_l2norm_fwd
from .recurrent import tt_fused_recurrent_gated_delta_rule

__all__ = [
    "tt_causal_conv1d_fn",
    "tt_causal_conv1d_update",
    "tt_chunk_gated_delta_rule",
    "tt_fused_gdn_gating",
    "tt_l2norm_fwd",
    "tt_fused_recurrent_gated_delta_rule",
]
