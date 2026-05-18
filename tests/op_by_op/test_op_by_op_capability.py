# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Test to verify that explorer build packages the op-by-op infra and its
transitive imports needed by tests/op_by_op/op_by_op_test.py.
"""

import op_by_op_infra
from op_by_op_infra.pydantic_models import OpTest, model_to_dict
from op_by_op_infra.workflow import execute_extracted_ops, extract_ops_from_module

print("op_by_op_infra imports OK")
