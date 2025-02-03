# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

from conftest import RecordProperties


def compile_fail(reason: str) -> str:
    return f"Compile failed: {reason}"


def runtime_fail(reason: str) -> str:
    return f"Runtime failed: {reason}"


def record_unary_op_test_properties(
    record_property: Callable, framework_op_name: str, op_name: str
):
    record_property(RecordProperties.OP_KIND.value, "Unary op")
    record_property(RecordProperties.FRAMEWORK_OP_NAME.value, framework_op_name)
    record_property(RecordProperties.OP_NAME.value, op_name)


def record_binary_op_test_properties(
    record_property: Callable, framework_op_name: str, op_name: str
):
    record_property(RecordProperties.OP_KIND.value, "Binary op")
    record_property(RecordProperties.FRAMEWORK_OP_NAME.value, framework_op_name)
    record_property(RecordProperties.OP_NAME.value, op_name)


def record_op_test_properties(
    record_property: Callable, op_kind: str, framework_op_name: str, op_name: str
):
    record_property(RecordProperties.OP_KIND.value, op_kind)
    record_property(RecordProperties.FRAMEWORK_OP_NAME.value, framework_op_name)
    record_property(RecordProperties.OP_NAME.value, op_name)


def record_model_test_properties(record_property: Callable, model_name: str):
    record_property(RecordProperties.MODEL_NAME.value, model_name)
