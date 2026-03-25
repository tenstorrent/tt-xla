// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "api/serialized_executable_instance.h"

// c++ standard library includes
#include <string>

// tt-xla includes
#include "api/buffer_instance.h"
#include "api/error_instance.h"
#include "api/module_builder/module_builder.h"

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
  const std::string &ttir_code = executable_image->getTTIRMlirCode();
  const std::string &ttnn_code = executable_image->getTTNNMlirCode();
  std::vector<std::byte> flatbuffer_data;
  if (executable_image->getCompileOptions().backend ==
      BackendRuntime::TTNNFlatbuffer) {
    const tt::runtime::Binary &flatbuffer_binary =
        static_cast<const FlatbufferExecutableImage *>(executable_image)
            ->getFlatbufferBinary();
    // TODO(sgligorijevic): We could avoid double copy if storeToMemory took a
    // span. Issue: https://github.com/tenstorrent/tt-mlir/issues/4822
    flatbuffer_binary.storeToMemory(flatbuffer_data);
  } else {
    flatbuffer_data.resize(0);
  }

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
