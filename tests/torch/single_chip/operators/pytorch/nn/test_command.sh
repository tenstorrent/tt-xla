# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Commands for running softmax tests


# Run single test with specific parameters:
pytest -svv test_softmax.py::test_softmax_single_params --softmax_model 'ModelFromAnotherOp' --softmax_input_shape_dim '((1, 1), 0)' --df 'Float16_b' --mf 'HiFi4' --runxfail --no-skips
pytest -svv test_softmax.py::test_softmax_single_params --softmax_model 'ModelFromHost' --softmax_input_shape_dim '(45, 17), 0' --df 'Float16_b' --mf 'HiFi4' --runxfail --no-skips
