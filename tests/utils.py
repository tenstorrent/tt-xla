# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


def compile_fail(reason: str) -> str:
    return f"Compile failed: {reason}"


def runtime_fail(reason: str) -> str:
    return f"Runtime failed: {reason}"
