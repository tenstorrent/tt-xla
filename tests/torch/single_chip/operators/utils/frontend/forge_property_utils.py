# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Loading Forge property utils for multiple components

from .detector import XLA_MODE


if XLA_MODE:
    from common.setup.frontend.xla.forge_property_utils import (
        forge_property_handler_var,
        ForgePropertyHandler,
        record_sweeps_test_tags,
        record_sweeps_expected_failing_reason,
        record_sweeps_detected_failing_reason,
    )
else:
    from forge.forge_property_utils import (
        forge_property_handler_var,
        ForgePropertyHandler,
        record_sweeps_test_tags,
        record_sweeps_expected_failing_reason,
        record_sweeps_detected_failing_reason,
    )


__all__ = [
    "forge_property_handler_var",
    "ForgePropertyHandler",
    "record_sweeps_test_tags",
    "record_sweeps_expected_failing_reason",
    "record_sweeps_detected_failing_reason",
]
