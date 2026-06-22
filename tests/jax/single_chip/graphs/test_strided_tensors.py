# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import numpy as np
import pytest
from utils import Category

# ---------- Fixtures ----------


@pytest.fixture
def cpu() -> jax.Device:
    return jax.devices("cpu")[0]


@pytest.fixture
def tt_device() -> jax.Device:
    return jax.devices("tt")[0]


# ---------- Helpers ----------


def _make_views() -> dict:
    """NumPy views exercising different (non-)contiguous memory layouts."""
    base = np.arange(32 * 16, dtype=np.float32).reshape(32, 16)
    return {
        "contiguous": base,
        "transpose": base.T,  # swapped strides
        "row_slice": base[::2],  # strided outer dim
        "col_slice": base[:, ::2],  # gap in inner dim
        "block": base[1:5, 2:7],  # offset sub-block
        "reversed": base[::-1],  # negative outer stride
    }


# ---------- Tests ----------


@pytest.mark.push
@pytest.mark.nightly
@pytest.mark.record_test_properties(category=Category.GRAPH_TEST)
@pytest.mark.parametrize("view_name", list(_make_views().keys()))
def test_strided_host_buffer_roundtrip(
    view_name: str, cpu: jax.Device, tt_device: jax.Device
):
    """
    Transferring a non-contiguous host buffer to the device and back must
    preserve its logical contents: the PJRT plugin has to gather the strided
    buffer into a contiguous tensor rather than reading it linearly.
    """
    view = _make_views()[view_name]
    expected = np.ascontiguousarray(view)

    on_device = jax.device_put(view, tt_device)
    got = np.asarray(jax.device_put(on_device, cpu))

    assert np.array_equal(got, expected), (
        f"'{view_name}' view was not reconstructed correctly:\n"
        f"expected:\n{expected}\n"
        f"got:\n{got}"
    )
