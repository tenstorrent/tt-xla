// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt


#include <utility>

#include "common/pjrt_implementation/error_instance.h"

#include "tt/runtime/runtime.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_UTILS_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_UTILS_H_
namespace tt::pjrt::utils {

PJRT_Buffer_Type
convertElementTypeToBufferType(tt::target::DataType ElementType);

std::pair<tt::target::DataType, size_t>
MapBufferTypeToElementType(PJRT_Buffer_Type buffer_type);

} // namespace tt::pjrt::utils

#endif
