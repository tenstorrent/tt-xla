# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Minimal public surface for infra.

Keep imports here narrow and backend-tolerant so torch/CUDA-only validation does not
eagerly require the full TT/XLA or JAX runtime stacks during collection.
"""

from .connectors.device_connector_factory import DeviceConnectorFactory
from .utilities.types import Framework, Model

try:
    from .evaluators.evaluation_config import ComparisonConfig
except Exception:
    ComparisonConfig = None

try:
    from .testers.single_chip.model.model_tester import RunMode
except Exception:
    RunMode = None

try:
    from .testers.single_chip.model.torch_model_tester import TorchModelTester
except Exception:
    TorchModelTester = None

try:
    from .testers.single_chip.model.jax_model_tester import JaxModelTester
except Exception:
    JaxModelTester = None

try:
    from .testers.graph.graph_test import (
        run_graph_test,
        run_graph_test_with_random_inputs,
    )
    from .testers.multichip.graph.jax_multichip_graph_tester import (
        run_jax_multichip_graph_test_with_random_inputs,
    )
    from .testers.multichip.op.jax_multichip_op_tester import (
        run_jax_multichip_op_test_with_random_inputs,
        serialize_jax_multichip_op,
        serialize_jax_multichip_op_with_random_inputs,
    )
    from .testers.op.op_test import run_op_test, run_op_test_with_random_inputs
except Exception:
    run_graph_test = None
    run_graph_test_with_random_inputs = None
    run_jax_multichip_graph_test_with_random_inputs = None
    run_jax_multichip_op_test_with_random_inputs = None
    serialize_jax_multichip_op = None
    serialize_jax_multichip_op_with_random_inputs = None
    run_op_test = None
    run_op_test_with_random_inputs = None
