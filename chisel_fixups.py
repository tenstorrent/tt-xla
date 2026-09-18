"""Pytest plugin: fix a Chisel golden bug and enable isolation mode.

1. ``chisel_ttnn_to_layout`` reads ``op.attributes["layout"]``, which our IR never
   carries (0 of 816 to_layout ops in the decoder graph). The attribute is unused
   by ``ttnn_to_layout_golden`` -- it only clones and casts to the result dtype --
   so the lookup is a pure bug that would leave every to_layout without a golden
   and silently re-seed the accumulated chain from device.
2. The ``chisel_context`` fixture builds a session with the default
   ``ChiselChecksConfig`` (isolation=False). Re-configure once the session exists.
"""
import os

import pytest


@pytest.fixture(autouse=True)
def chisel_fixups(request, chisel_context):
    if not request.config.getoption("--enable-chisel", default=False):
        yield
        return

    import chisel
    from chisel.validators import ChiselChecksConfig
    from golden import mapping as gm
    import ttmlir.dialects.ttnn as ttnn_d

    def _to_layout(op, inputs):
        return gm.ttnn_to_layout_golden(
            input_tensor=inputs["input"],
            layout_attr=None,  # unused by the golden
            output_type_mlir=op.results[0].type.element_type,
        )

    gm.CHISEL_GOLDEN_MAPPINGS[ttnn_d.ToLayoutOp] = _to_layout

    # The MoE compute chain had NO usable golden: chisel_ttnn_sparse_matmul and
    # chisel_ttnn_all_to_all_combine call `sparse_matmul_golden` /
    # `all_to_all_combine_golden`, which do not exist -- the real functions are
    # named ttir_*. They also pass pre-unpacked values, while the real goldens
    # take raw MLIR attributes plus output_type_mlir. Result upstream: 90
    # chisel_bug records and zero coverage of the ops that do the MoE math,
    # across all 30 layers.
    def _sparse_matmul(op, inputs):
        return gm.ttir_sparse_matmul_golden(
            a=inputs["a"],
            b=inputs["b"],
            sparsity=inputs["sparsity"],
            is_input_a_sparse_attr=op.attributes["is_input_a_sparse"],
            is_input_b_sparse_attr=op.attributes["is_input_b_sparse"],
            nnz_attr=gm._attr_get_value(op.attributes, "nnz"),
            output_type_mlir=op.results[0].type.element_type,
        )

    def _all_to_all_combine(op, inputs):
        return gm.ttir_all_to_all_combine_golden(
            input_tensor=inputs["input_tensor"],
            expert_metadata=inputs["expert_metadata"],
            expert_mapping=inputs["expert_mapping"],
            num_devices_attr=op.attributes["num_devices"],
            cluster_axis_attr=op.attributes["cluster_axis"],
            num_experts_per_tok_attr=op.attributes["num_experts_per_tok"],
            output_type_mlir=op.results[0].type.element_type,
        )

    gm.CHISEL_GOLDEN_MAPPINGS[ttnn_d.SparseMatmulOp] = _sparse_matmul
    gm.CHISEL_GOLDEN_MAPPINGS[ttnn_d.AllToAllCombineOp] = _all_to_all_combine

    # Accumulation keeps a host-side golden pool holding every live SSA until it
    # is deallocated, so throughput decays monotonically as the pool grows
    # (measured 13603 -> 525 -> 38 records/100s on the same model). Isolation
    # only stashes the current op's inputs and runs at a flat rate. Default to
    # isolation; opt in with CHISEL_MODES=acc or both.
    modes = os.environ.get("CHISEL_MODES", "iso").lower()
    chisel.configure(
        checks_config=ChiselChecksConfig(
            isolation=modes in ("iso", "both"),
            accumulation=modes in ("acc", "both"),
        )
    )
    yield
