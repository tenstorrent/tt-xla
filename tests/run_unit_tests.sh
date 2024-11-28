#!/bin/bash/

source venv/activate

pytest -svv tests/ops/test_add.py::test_add
pytest -svv tests/graphs/test_arbitrary_op_chain.py::test_arbitrary_op_chain
pytest -svv tests/models/test_simple_nn.py::test_simple_nn
