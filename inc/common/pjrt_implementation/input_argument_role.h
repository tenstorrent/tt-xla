// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_INPUT_ARGUMENT_ROLE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_INPUT_ARGUMENT_ROLE_H_

namespace tt::pjrt {

// Enum to represent the role of input arguments
enum class InputArgumentRole {
  kInput, // Regular input data
  kWeight // Weight/parameter data
};

} // namespace tt::pjrt

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_INPUT_ARGUMENT_ROLE_H_
