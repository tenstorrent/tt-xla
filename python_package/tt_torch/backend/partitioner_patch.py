# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Workaround for torch_xla's CPU-fallback graph partitioner choking on
partitions that have no outputs.

When a compiled region mutates a tensor in place and discards the result
(e.g. a vLLM GatedDeltaNet layer writing its conv/ssm state back into the KV
cache via ``aten.copy_``), functionalization turns that into a ``copy_`` node
with ``num_users == 0``. torch_xla's ``CapabilityBasedPartitioner`` can group
such a sink (plus the compute that feeds only it) into its own partition whose
nodes have no users *outside* the partition. That partition has zero FX
outputs, and ``torch.fx``'s fuser then trips:

    assert last_output_node is not None   # fx/passes/utils/fuser_utils.py

because it looks for a partition output node and finds none.

Buffer-donor hints (``dynamo_set_buffer_donor_``) do not help: the partitioner
decides outputs purely from FX ``num_users`` edges, and a write-back into an
input placeholder always has zero users.

Fix: drop output-less partitions before fusing. Their nodes then stay in the
parent graph and execute via the normal (lazy-XLA) path instead of being fused
into an explicitly extracted TT subgraph. They still run on device; they are
just not packed into a fused module. This keeps the in-place mutation intact
and avoids both the assert and the degenerate "fused module with no real
output" state that downstream torch_xla code (``InputCollector`` /
``extract_internal``, which expects every fused submodule to be executed and
tagged with ``xla_args``) cannot handle.
"""

from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner

_PATCH_FLAG = "_tt_dropped_outputless_partitions_patched"


def _has_external_output(partition) -> bool:
    """True if any node in ``partition`` has a user outside the partition.

    Mirrors how ``fuse_as_graphmodule`` computes a partition's outputs: a node
    is an output iff at least one of its users is not in the partition. A
    partition where every node's users are all internal (or a terminal node has
    no users at all, e.g. an in-place ``copy_`` sink) has zero outputs.
    """
    node_set = set(partition.nodes)
    return any(
        any(user not in node_set for user in node.users)
        for node in partition.nodes
    )


def install_partitioner_outputless_partition_workaround() -> None:
    """Idempotently patch ``CapabilityBasedPartitioner.propose_partitions`` to
    drop partitions that would have no outputs."""
    if getattr(CapabilityBasedPartitioner, _PATCH_FLAG, False):
        return

    _orig_propose_partitions = CapabilityBasedPartitioner.propose_partitions

    def propose_partitions(self):
        partitions = _orig_propose_partitions(self)
        return [p for p in partitions if _has_external_output(p)]

    CapabilityBasedPartitioner.propose_partitions = propose_partitions
    setattr(CapabilityBasedPartitioner, _PATCH_FLAG, True)
