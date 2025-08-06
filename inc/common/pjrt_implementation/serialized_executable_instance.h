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
#include <cstddef>

#ifndef TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_SERIALIZED_EXECUTABLE_INSTANCE_H_
#define TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_SERIALIZED_EXECUTABLE_INSTANCE_H_

namespace tt::pjrt {

struct SerializationHeader {
  // Serialization format:
  // Magic string: "TTSERv00", 8 bytes
  // Header: 48 bytes, 3*2*sizeof(u64)
  // Offset + size for TTIR, TTNN and FB
  // Body:
  // TTIR, variable
  // TTNN, variable
  // Flatbuffer, variable
  // Total size: 8 + 48 + TTIR.size() + TTNN.size() + flatbuffer_data.size()

  char magic[8];
  uint64_t ttir_offset;
  uint64_t ttir_size;
  uint64_t ttnn_offset;
  uint64_t ttnn_size;
  uint64_t fb_offset;
  uint64_t fb_size;

  SerializationHeader(const std::string &ttir_code,
                      const std::string &ttnn_code,
                      const std::vector<std::byte> &flatbuffer_data)
      : ttir_offset(0), ttir_size(ttir_code.size()),
        ttnn_offset(ttir_offset + ttir_size), ttnn_size(ttnn_code.size()),
        fb_offset(ttnn_offset + ttnn_size), fb_size(flatbuffer_data.size()) {
    std::memcpy(magic, "TTSERv00", sizeof(magic));
  }

  size_t getHeaderSize() const { return sizeof(magic) + 3 * sizeof(uint64_t); }
  size_t getPayloadSize() const {
    return getHeaderSize() + ttir_size + ttnn_size + fb_size;
  }
};

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

  // Returns the serialized payload data.
  const std::vector<std::byte> &getSerializedPayload() const {
    return m_payload;
  }

  // Gets the size of the serialized payload in bytes.
  size_t getSerializedPayloadSizeInBytes() const { return m_payload.size(); }

private:
  // Creates serialized executable instance from the executable image.
  SerializedExecutableInstance(const ExecutableImage *executable_image);

  std::byte *writeRaw(std::byte *dst, const void *src, size_t size) {
    std::memcpy(dst, src, size);
    return dst + size;
  }

  template <typename T> std::byte *write(std::byte *dst, const T &src) {
    return writeRaw(dst, &src, sizeof(src));
  }

  // Serialized flatbuffer binary data.
  std::vector<std::byte> m_payload;
};

} // namespace tt::pjrt

#endif // TT_XLA_INC_COMMON_PJRT_IMPLEMENTATION_SERIALIZED_EXECUTABLE_INSTANCE_H_
