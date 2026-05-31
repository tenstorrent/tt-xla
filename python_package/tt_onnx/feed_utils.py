# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Coerce ONNX session feeds to model input element types."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import onnx
from onnx import TensorProto


def onnx_elem_type_to_numpy(elem_type: int) -> np.dtype:
    mapping: dict[int, np.dtype] = {
        TensorProto.FLOAT: np.dtype(np.float32),
        TensorProto.DOUBLE: np.dtype(np.float64),
        TensorProto.INT32: np.dtype(np.int32),
        TensorProto.INT64: np.dtype(np.int64),
        TensorProto.UINT32: np.dtype(np.uint32),
        TensorProto.UINT64: np.dtype(np.uint64),
        TensorProto.BOOL: np.dtype(np.bool_),
    }
    if elem_type not in mapping:
        raise TypeError(f"Unsupported ONNX element type: {elem_type}")
    return mapping[elem_type]


def input_elem_types(model: onnx.ModelProto) -> dict[str, int]:
    initializer_names = {init.name for init in model.graph.initializer}
    types: dict[str, int] = {}
    for value_info in model.graph.input:
        if value_info.name in initializer_names:
            continue
        types[value_info.name] = value_info.type.tensor_type.elem_type
    return types


def prepare_feed(
    model: onnx.ModelProto,
    feed: Mapping[str, np.ndarray | list | tuple],
) -> dict[str, np.ndarray]:
    """Cast feed values to dtypes declared in the ONNX graph inputs."""
    elem_types = input_elem_types(model)
    prepared: dict[str, np.ndarray] = {}
    for name, value in feed.items():
        if name not in elem_types:
            raise KeyError(f"Unknown ONNX input {name!r}; expected one of {sorted(elem_types)}")
        dtype = onnx_elem_type_to_numpy(elem_types[name])
        prepared[name] = np.asarray(value, dtype=dtype)
    return prepared


def prepare_feed_for_ort(
    ort_session,
    feed: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Cast feed values to dtypes expected by ONNX Runtime inputs."""
    type_by_name = {inp.name: inp.type for inp in ort_session.get_inputs()}
    prepared: dict[str, np.ndarray] = {}
    for name, value in feed.items():
        ort_type = type_by_name.get(name, "tensor(float)")
        if "int64" in ort_type:
            dtype = np.int64
        elif "int32" in ort_type:
            dtype = np.int32
        elif "double" in ort_type:
            dtype = np.float64
        elif "bool" in ort_type:
            dtype = np.bool_
        else:
            dtype = np.float32
        prepared[name] = np.asarray(value, dtype=dtype)
    return prepared
