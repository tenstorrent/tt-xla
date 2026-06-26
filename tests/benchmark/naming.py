# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Identifier / string helpers: sanitizing names and building the export and
display names used for compiled-module export and dashboard reporting.

Pure and dependency-free.
"""

import re
import secrets
from typing import Any, Optional, Union


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be safe for use in filenames.
    Replaces illegal filesystem characters with underscores and converts to lowercase.
    """
    # Replace illegal filesystem characters: / \ : * ? " < > | and spaces
    # Also replace dots and dashes for consistency
    sanitized = re.sub(r'[/\\:*?"<>|\s.\-]', "_", str(name))
    # Remove consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores and convert to lowercase
    return sanitized.strip("_").lower()


def sanitize_model_name(value: Any) -> str:
    text = str(value).strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-zA-Z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip("_").lower()
    return text or "na"


def build_xla_export_name(
    model_name: str,
    num_layers: Optional[Union[int, str]],
    batch_size: int,
    input_sequence_length: Optional[int],
) -> str:
    """Build a standardized export name for XLA benchmark runs."""
    run_id = secrets.token_hex(2)

    if num_layers is None or (isinstance(num_layers, int) and num_layers <= 0):
        layers_part = None
    else:
        layers_part = f"{num_layers}lyr"

    if not isinstance(model_name, str) and hasattr(model_name, "name"):
        model_name = model_name.name
    parts = [sanitize_model_name(model_name)]
    if layers_part:
        parts.append(layers_part)
    parts.append(f"bs{batch_size}")
    if input_sequence_length is not None and input_sequence_length > 0:
        parts.append(f"isl{input_sequence_length}")
    parts.append(f"run{run_id}")
    return "_".join(parts)


def resolve_display_name(request: Any = None, fallback: Optional[str] = None) -> str:
    """Resolve a display name, optionally overriding with pytest test name."""
    name = None
    if (
        request is not None
        and hasattr(request, "node")
        and hasattr(request.node, "name")
    ):
        test_name = request.node.name
        if test_name and test_name.startswith("test_"):
            name = test_name[5:]

    if not name:
        name = sanitize_model_name(fallback or "")
    return name
