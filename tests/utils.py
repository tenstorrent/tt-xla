# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from typing import Callable

import jax
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


@contextmanager
def enable_x64():
    """
    Context manager that temporarily enables x64 in jax.config.

    Isolated as a context manager so that it doesn't change global config for all jax
    imports and cause unexpected fails elsewhere.
    """
    try:
        # Set the config to True within this block, and yield back control.
        jax.config.update("jax_enable_x64", True)
        yield
    finally:
        # After `with` statement ends, turn it off again.
        jax.config.update("jax_enable_x64", False)
