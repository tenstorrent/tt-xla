// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/pjrt_implementation/serialized_executable_instance.h"

// tt-xla includes
#include "common/pjrt_implementation/buffer_instance.h"
#include "common/pjrt_implementation/error_instance.h"

namespace tt::pjrt {

std::unique_ptr<SerializedExecutableInstance>
SerializedExecutableInstance::createInstance(
    const ExecutableImage *executable_image) {
  struct make_unique_enabler : public SerializedExecutableInstance {
    make_unique_enabler(const ExecutableImage *executable_image)
        : SerializedExecutableInstance(executable_image) {}
  };
  return std::make_unique<make_unique_enabler>(std::move(executable_image));
}

SerializedExecutableInstance::SerializedExecutableInstance(
    const ExecutableImage *executable_image)
    : m_ttir_code(executable_image->getTTIRMlirCode()) {
  const tt::runtime::Binary &flatbuffer_binary =
      executable_image->getFlatbufferBinary();
  flatbuffer_binary.storeToMemory(m_serialized_flatbuffer);
}

} // namespace tt::pjrt
