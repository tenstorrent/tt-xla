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

// This is a bare implementation throwing UNDEFINED errors. This way new
// functions will not segmentation fault on invocation.
void bindUndefineds(PJRT_Api *api);

// Binds all the PJRT API functions implementations.
void bindApi(PJRT_Api *api);

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_BINDINGS_H_
