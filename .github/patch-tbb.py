#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Patches the system Intel TBB task.h header before the manylinux wheel build.
#
# Problem: the manylinux container has GCC 15's <execution> header, which
# unconditionally routes std::execution through pstl/parallel_backend_tbb.h,
# which includes /usr/include/tbb/task.h. That header has an in-class static
# const initializer that casts an out-of-range integer to an enum type:
#
#   static const kind_type binding_completed = kind_type(bound+1);  // 2, range [0,1]
#
# Clang 17+ rejects this as a non-constant expression (-Wenum-constexpr-conversion),
# which breaks the Tracy profiler tool build inside tt-metal. The system TBB
# package predates this enforcement and is not our code to fix upstream, so we
# wrap the offending line with Clang diagnostic pragmas before building.

import pathlib
import sys

TBB_TASK_H = pathlib.Path("/usr/include/tbb/task.h")
NEEDLE = "    static const kind_type binding_completed = kind_type(bound+1);"
PATCH = (
    "#pragma clang diagnostic push\n"
    '#pragma clang diagnostic ignored "-Wenum-constexpr-conversion"\n'
    + NEEDLE
    + "\n"
    "#pragma clang diagnostic pop"
)

if not TBB_TASK_H.exists():
    print(f"patch-tbb.py: {TBB_TASK_H} not found, skipping")
    sys.exit(0)

text = TBB_TASK_H.read_text()

if NEEDLE not in text:
    print(f"patch-tbb.py: {TBB_TASK_H} already patched or line not found, skipping")
    sys.exit(0)

TBB_TASK_H.write_text(text.replace(NEEDLE, PATCH))
print(f"patch-tbb.py: patched {TBB_TASK_H}")
