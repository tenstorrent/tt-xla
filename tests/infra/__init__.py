# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Exposes only what is really needed to write tests, nothing else."""

# isort: off
import sys

# Pytest and most tests add ``tests/`` on ``sys.path`` and import this tree as
# ``infra``. One-off commands often use ``import tests.infra``, which would otherwise
# load the same ``tests/infra`` directory a second time as top-level ``infra`` when
# in-repo code does ``from infra...``. That duplicate package breaks with a circular
# import; register one module object under both names.
if __name__ == "tests.infra":
    sys.modules.setdefault("infra", sys.modules[__name__])

# NOTE: Import order matters here - evaluators must come before connectors
# to avoid circular import (connectors -> utilities -> runners -> connectors)
from .evaluators import ComparisonConfig
from .connectors import DeviceConnectorFactory, JaxDeviceConnector

# isort: on
from .testers import (
    JaxModelTester,
    RunMode,
    TorchModelTester,
    run_graph_test,
    run_graph_test_with_random_inputs,
    run_jax_multichip_graph_test_with_random_inputs,
    run_jax_multichip_op_test_with_random_inputs,
    run_op_test,
    run_op_test_with_random_inputs,
    serialize_jax_multichip_op,
    serialize_jax_multichip_op_with_random_inputs,
)
from .utilities import (
    Framework,
    MLACache,
    MLAStaticLayer,
    Model,
    ShardingMode,
    enable_shardy,
    initialize_flax_linen_parameters_on_cpu,
    make_flax_linen_parameters_partition_specs_on_cpu,
    make_partition_spec,
    random_image,
    random_tensor,
)
from .workloads import JaxMultichipWorkload, Workload
