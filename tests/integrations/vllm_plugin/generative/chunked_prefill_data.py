# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared prompt data for the multichip chunked-prefill tests."""

# A prompt comfortably longer than a small prefill_chunk_size (e.g. 128) so the
# prompt splits into several block-aligned chunks and chunks 2..N exercise the
# cached-prefix chunked-SDPA path (tt-xla #4986/#5691). Shared by the multichip
# chunked-prefill tests.
CHUNKED_PREFILL_PARA = (
    "The history of computing spans many centuries. Early mechanical "
    "calculators gave way to electromechanical machines, and then to the "
    "electronic digital computers that define the modern era. Each generation "
    "brought dramatic improvements in speed, reliability, and cost. "
)
CHUNKED_PREFILL_PROMPT = (
    "Continue in English. " + (CHUNKED_PREFILL_PARA * 5) + "\nContinuation:"
)
