# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Name helpers: sanitizing identifiers and building the export / display names
used for compiled-module export and dashboard reporting. Pure and dependency-free.
"""

import re
import secrets
from typing import Any, Optional, Union


def sanitize_name(value: Any) -> str:
    """Lowercase ``value``, collapsing each run of non-alphanumerics to one underscore.

    Safe for both filenames and dashboard names.
    Empty input becomes ``"na"``.
    """
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip())
    return text.strip("_").lower() or "na"


def build_xla_export_name(
    model_name: str,
    num_layers: Optional[Union[int, str]],
    batch_size: int,
    input_sequence_length: Optional[int],
) -> str:
    """Build the export name ``<model>[_<N>lyr]_bs<N>[_isl<N>]_run<hex>``.

    The layer and input-sequence-length parts are omitted when not set.
    ``run`` is a random suffix so repeated runs export to distinct names.
    """
    if not isinstance(model_name, str) and hasattr(model_name, "name"):
        model_name = model_name.name

    parts = [sanitize_name(model_name)]
    if num_layers and not (isinstance(num_layers, int) and num_layers <= 0):
        parts.append(f"{num_layers}lyr")
    parts.append(f"bs{batch_size}")
    if input_sequence_length and input_sequence_length > 0:
        parts.append(f"isl{input_sequence_length}")
    parts.append(f"run{secrets.token_hex(2)}")
    return "_".join(parts)


def perf_metrics_filename(display_name: str) -> str:
    """Base name for the per-graph TTNN perf-metric files emitted by the compiler.

    Single source of truth for the convention shared by every driver that
    aggregates perf metrics (vision / encoder / imagegen / llm / resnet).
    """
    return f"tt_xla_{display_name}_perf_metrics"


def resolve_display_name(request: Any = None, fallback: Optional[str] = None) -> str:
    """Display name from the pytest test name (``test_`` stripped), else fallback."""
    node = getattr(request, "node", None)
    test_name = getattr(node, "name", "") if node is not None else ""
    if test_name.startswith("test_"):
        return test_name[5:]
    return sanitize_name(fallback or "")
