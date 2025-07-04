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
  createInstance(const ExecutableImage *executable_image);

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
  const std::vector<std::byte> &getSerializedFlatbuffer() const {
    return m_serialized_flatbuffer;
  }

  // Gets the size of the serialized flatbuffer in bytes.
  size_t getSerializedSizeInBytes() const {
    return m_serialized_flatbuffer.size();
  }

  // Gets the TTIR code of the executable image.
  const std::string &getTTIRCode() const { return m_ttir_code; }

private:
  // Creates serialized executable instance from the executable image.
  SerializedExecutableInstance(const ExecutableImage *executable_image);

  // Serialized flatbuffer binary data.
  std::vector<std::byte> m_serialized_flatbuffer;

  // TTIR representation of the executable image.
  const std::string m_ttir_code;
};

} // namespace tt::pjrt

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_SERIALIZED_EXECUTABLE_INSTANCE_H_
