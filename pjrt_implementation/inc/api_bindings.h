// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_BINDINGS_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_BINDINGS_H_

namespace tt::pjrt {

// Binds all the PJRT API functions implementations.
void bindApi(PJRT_Api *api);

// This is a bare implementation throwing UNDEFINED errors. This way new
// functions will not segmentation fault on invocation.
void bindUndefineds(PJRT_Api *api);

} // namespace tt::pjrt

// C API for accessing pipeline stage information from Python
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Get the current pipeline stage as a string.
 * Returns a static string representing the current stage.
 * Thread-safe operation.
 *
 * The returned string is one of:
 * - "NOT_STARTED"
 * - "FE_COMPILATION_START"
 * - "TTMLIR_COMPILATION_START"
 * - "RUNTIME_EXECUTION_START"
 * - "PCC_COMPARISON_START"
 * - "PCC_COMPARISON_PASSED"
 * - "UNKNOWN"
 */
const char *PJRT_TT_GetCurrentPipelineStage();

/**
 * Set the current pipeline stage by name (for Python-side stage tracking).
 * Thread-safe operation.
 *
 * Accepts stage name strings:
 * - "FE_COMPILATION_START"
 * - "TTMLIR_COMPILATION_START"
 * - "RUNTIME_EXECUTION_START"
 * - "PCC_COMPARISON_START"
 * - "PCC_COMPARISON_PASSED"
 */
void PJRT_TT_SetPipelineStageByName(const char *stage_name);

#ifdef __cplusplus
}
#endif

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_BINDINGS_H_
