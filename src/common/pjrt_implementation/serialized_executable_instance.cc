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

// c++ standard library includes
#include <string>

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
    const ExecutableImage *executable_image) {
  const tt::runtime::Binary &flatbuffer_binary =
      executable_image->getFlatbufferBinary();
  const std::string &ttir_code = executable_image->getTTIRMlirCode();
  const std::string &ttnn_code = executable_image->getTTNNMlirCode();

  std::vector<std::byte> flatbuffer_data;
  // TODO(stefan): We could avoid double copy if storeToMemory took a span.
  flatbuffer_binary.storeToMemory(flatbuffer_data);

  SerializationHeader header(ttir_code, ttnn_code, flatbuffer_data);
  m_payload.resize(header.getPayloadSize());

  std::byte *data_ptr = m_payload.data();

  data_ptr = write(data_ptr, header);
  data_ptr = writeRaw(data_ptr, ttir_code.data(), ttir_code.size());
  data_ptr = writeRaw(data_ptr, ttnn_code.data(), ttnn_code.size());
  data_ptr = writeRaw(data_ptr, flatbuffer_data.data(), flatbuffer_data.size());

  // assert that we wrote the correct number of bytes
  assert(data_ptr == m_payload.data() + m_payload.size());
}

} // namespace tt::pjrt
