# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from .graph.jax_multichip_graph_tester import (
    run_jax_multichip_graph_test_with_random_inputs,
)
from .model.jax_multichip_model_tester import JaxMultichipModelTester
from .op.jax_multichip_op_tester import run_jax_multichip_op_test_with_random_inputs
