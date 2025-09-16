# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import threading
import os

# Global state for tracking current model
_current_model = None
_file_lock = threading.Lock()
_log_file = "decomposition_log.txt"


def set_current_model(model_name: str):
    """Set the current model name for decomposition logging."""
    global _current_model
    _current_model = model_name

    # Write model header to log file
    with _file_lock:
        try:
            with open(_log_file, "a") as f:
                f.write(f"\n=== MODEL: {model_name} ===\n")
        except Exception:
            pass  # Ignore file writing errors


def log_decomposition(operation_type: str, op_name: str):
    """Log a decomposition operation for the current model."""
    with _file_lock:
        try:
            with open(_log_file, "a") as f:
                f.write(f"{operation_type}: {op_name}\n")
        except Exception:
            pass  # Ignore file writing errors


def clear_log():
    """Clear the decomposition log file."""
    try:
        if os.path.exists(_log_file):
            os.remove(_log_file)
    except Exception:
        pass
