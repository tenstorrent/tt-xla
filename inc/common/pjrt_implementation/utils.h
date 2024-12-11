// SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>

#include "common/pjrt_implementation/error_instance.h"
#include "tt/runtime/runtime.h"

#ifndef TT_XLA_UTILS_H_
#define TT_XLA_UTILS_H_
namespace tt::pjrt::utils {

PJRT_Buffer_Type
convertElementTypeToBufferType(tt::target::DataType ElementType);

std::pair<tt::target::DataType, size_t>
MapBufferTypeToElementType(PJRT_Buffer_Type buffer_type);

} // namespace tt::pjrt::utils

#endif
