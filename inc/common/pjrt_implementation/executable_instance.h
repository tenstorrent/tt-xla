// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// c++ standard library includes
#include <memory>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-xla includes
#include "common/pjrt_implementation/executable_image.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EXECUTABLE_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EXECUTABLE_INSTANCE_H_

namespace tt::pjrt {

// Represents `PJRT_Executable` structure and the functionality around it. It
// knows how to take a compiled executable image and execution options and
// serialize/deserialize them so an executable can be stored and loaded as
// needed.
class ExecutableInstance {
public:
  // Creates new executable instance.
  static std::unique_ptr<ExecutableInstance>
  createInstance(std::shared_ptr<ExecutableImage> executable_image);

  // Binds PJRT API functions implementation related to PJRT_Buffer structure.
  static void bindApi(PJRT_Api *api);

  // Casts this executable instance to PJRT_Executable pointer.
  operator PJRT_Executable *() {
    return reinterpret_cast<PJRT_Executable *>(this);
  }

  // Casts the PJRT_Executable pointer to ExecutableInstance pointer.
  static ExecutableInstance *unwrap(PJRT_Executable *exe) {
    return reinterpret_cast<ExecutableInstance *>(exe);
  }

  // TODO_OOM: Check if can be const.
  // Returns pointer to the underlying executable image.
  ExecutableImage *getExecutableImage() { return m_executable_image.get(); }

private:
  // Constructs executable instance from the compiled executable image.
  ExecutableInstance(std::shared_ptr<ExecutableImage> executable_image)
      : m_executable_image(std::move(executable_image)) {}

  // Executable image which is shared between executable and loaded executable
  // instances.
  std::shared_ptr<ExecutableImage> m_executable_image;
};

namespace internal {

// TODO_OOM: finish

} // namespace internal

} // namespace tt::pjrt

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_EXECUTABLE_INSTANCE_H_
