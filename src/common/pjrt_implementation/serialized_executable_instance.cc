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
#include <string>

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

  // Serialization format:
  // Magic string: "TTSERv00", 8 bytes
  // Offset + size for TTIR, TTNN and FB, 3*2*8= 48 bytes
  // TTIR, variable
  // TTNN, variable
  // Flatbuffer, variable
  // Total size: 8 + 48 + TTIR.size() + TTNN.size() + flatbuffer_data.size()
  size_t total_size =
      8 + 48 + ttir_code.size() + ttnn_code.size() + flatbuffer_data.size();

  m_payload.resize(total_size);
  std::byte *data_ptr = m_payload.data();
  // Write magic string.
  std::string magic_string = "TTSERv00";
  std::memcpy(data_ptr, magic_string.data(), magic_string.size());
  data_ptr += magic_string.size();
  // Offsets are calculated from the start of the data section,
  // ie we expect TTIR offset to be 0 almost always.
  uint64_t ttir_offset = 0;
  uint64_t ttir_size = ttir_code.size();
  uint64_t ttnn_offset = ttir_offset + ttir_size;
  uint64_t ttnn_size = ttnn_code.size();
  uint64_t fb_offset = ttnn_offset + ttnn_size;
  uint64_t fb_size = flatbuffer_data.size();
  // Write offsets and sizes.
  std::memcpy(data_ptr, &ttir_offset, sizeof(ttir_offset));
  data_ptr += sizeof(ttir_offset);
  std::memcpy(data_ptr, &ttir_size, sizeof(ttir_size));
  data_ptr += sizeof(ttir_size);
  std::memcpy(data_ptr, &ttnn_offset, sizeof(ttnn_offset));
  data_ptr += sizeof(ttnn_offset);
  std::memcpy(data_ptr, &ttnn_size, sizeof(ttnn_size));
  data_ptr += sizeof(ttnn_size);
  std::memcpy(data_ptr, &fb_offset, sizeof(fb_offset));
  data_ptr += sizeof(fb_offset);
  std::memcpy(data_ptr, &fb_size, sizeof(fb_size));
  data_ptr += sizeof(fb_size);
  // Write TTIR code.
  std::memcpy(data_ptr, ttir_code.data(), ttir_size);
  data_ptr += ttir_size;
  // Write TTNN code.
  std::memcpy(data_ptr, ttnn_code.data(), ttnn_size);
  data_ptr += ttnn_size;
  // Write flatbuffer data.
  std::memcpy(data_ptr, flatbuffer_data.data(), fb_size);
  data_ptr += fb_size;

  // assert that we wrote the correct number of bytes
  assert(data_ptr == m_payload.data() + m_payload.size());
}

} // namespace tt::pjrt
