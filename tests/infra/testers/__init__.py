# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal tester exports."""

try:
    from .base_tester import BaseTester
except Exception:
    BaseTester = None

try:
    from .single_chip.model.jax_model_tester import JaxModelTester
    from .single_chip.model.model_tester import RunMode
    from .single_chip.model.torch_model_tester import TorchModelTester
except Exception:
    RunMode = None
    TorchModelTester = None
    JaxModelTester = None

try:
    from .multichip.graph.jax_multichip_graph_tester import (
        run_jax_multichip_graph_test_with_random_inputs,
    )
    from .multichip.op.jax_multichip_op_tester import (
        run_jax_multichip_op_test_with_random_inputs,
        serialize_jax_multichip_op,
        serialize_jax_multichip_op_with_random_inputs,
    )
    from .single_chip.graph.graph_tester import (
        run_graph_test,
        run_graph_test_with_random_inputs,
    )
    from .single_chip.op.op_tester import run_op_test, run_op_test_with_random_inputs
except Exception:
    run_graph_test = None
    run_graph_test_with_random_inputs = None
    run_op_test = None
    run_op_test_with_random_inputs = None
    run_jax_multichip_graph_test_with_random_inputs = None
    run_jax_multichip_op_test_with_random_inputs = None
    serialize_jax_multichip_op = None
    serialize_jax_multichip_op_with_random_inputs = None
