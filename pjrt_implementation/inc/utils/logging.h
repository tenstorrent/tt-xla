// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// Loguru doesn't have debug named verbosity and the next verbosity after INFO
// is called `Verbosity_1`, so defining this to avoid doing `DLOG_F(1, ...)`.
#define LOG_DEBUG 1
#define LOG_VERBOSE 2

#include <loguru/loguru.hpp>

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_LOGGING_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_LOGGING_H_

namespace tt::pjrt {
void InitializeLogging();
} // namespace tt::pjrt

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_LOGGING_H_
