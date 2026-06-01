# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CPU golden for layer-0 LN+attn — same module as tt-xla codegen."""

from cpu_reference.forward import run_forward_from_fixtures
from cpu_reference.layer0_cpu import compare_layer0_ln_attn_stages

__all__ = ["run_forward_from_fixtures", "compare_layer0_ln_attn_stages"]
