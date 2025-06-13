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

// tt-mlir includes
#include "tt/runtime/runtime.h"

// tt-xla includes
#include "common/pjrt_implementation/device_instance.h"
#include "common/pjrt_implementation/executable_instance.h"
#include "common/status.h"

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_SERIALIZED_EXECUTABLE_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_SERIALIZED_EXECUTABLE_INSTANCE_H_

namespace tt::pjrt {

class SerializedExecutableInstance {
public:
  // Creates new serialized executable instance from the executable image.
  static std::unique_ptr<SerializedExecutableInstance>
  createInstance(std::shared_ptr<ExecutableImage> executable_image);

  // Binds PJRT API functions implementation related to
  // PJRT_SerializedExecutable structure.
  static void bindApi(PJRT_Api *api);

  // Casts this serialized executable instance to PJRT_SerializedExecutable
  // pointer.
  operator PJRT_SerializedExecutable *() {
    return reinterpret_cast<PJRT_SerializedExecutable *>(this);
  }

  // Casts the PJRT_SerializedExecutable pointer to PJRT_SerializedExecutable
  // pointer.
  static SerializedExecutableInstance *
  unwrap(PJRT_SerializedExecutable *executable) {
    return reinterpret_cast<SerializedExecutableInstance *>(executable);
  }

  // Returns the serialized flatbuffer data.
  const std::vector<std::byte> &getSerializedCode() const {
    return m_serialized_code;
  }

  // Releases the resources of this serialized executable instance.
  void releaseResources();

  // Gets the size of the serialized flatbuffer in bytes.
  size_t getSerializedSizeInBytes() const {
    return m_serialized_code.size();
  }

private:
  // Creates serialized executable instance from the executable image.
  SerializedExecutableInstance(
      std::shared_ptr<ExecutableImage> executable_image);

  // Executable image instance.
  std::shared_ptr<ExecutableImage> m_executable_image;

  // Serialized data buffer - best options:
  std::vector<std::byte> m_serialized_code;
};

namespace internal {} // namespace internal

} // namespace tt::pjrt

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_LOADED_EXECUTABLE_INSTANCE_H_
