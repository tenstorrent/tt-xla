// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

// c++ standard library includes
#include <cstddef>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-mlir includes
#include "tt/runtime/runtime.h"

// tt-xla includes
#include "api/device_instance.h"
#include "api/executable_instance.h"
#include "utils/status.h"

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_SERIALIZED_EXECUTABLE_INSTANCE_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_SERIALIZED_EXECUTABLE_INSTANCE_H_

namespace tt::pjrt {

struct SerializationHeader {
  // Serialization format:
  //   1. Header, 56 bytes
  //    1.1. Magic string: "TTSERv00", 8 bytes
  //    1.2  (Offset, Size) pairs for each section in data, 48 bytes
  //     1.2.1. Offset + size for TTIR, 2*sizeof(u64)
  //     1.2.2. Offset + size for TTNN, 2*sizeof(u64)
  //     1.2.3. Offset + size for Flatbuffer, 2*sizeof(u64)
  //   2. Body: variable size
  //     2.1. TTIR, variable size
  //     2.2. TTNN, variable size
  //     2.3. Flatbuffer, variable size
  // The offsets are relative to the start of the body.
  // Subsections in 2. are **NOT** assumed to be contiguous.
  // However the current serializer does pack them contiguously.
  // Subsections in 2. are assumed to be in the order defined above.
  // Total size: 56: + (flatbuffer_offset + flatbuffer_size)
  // This struct contains only the header part.

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

  size_t getHeaderSize() const { return sizeof(*this); }

  size_t getPayloadSize() const {
    return getHeaderSize() + ttir_size + ttnn_size +
           fb_size; // only correct because we pack them contiguously
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

  // Helper function that fills the buffer with data and advances the pointer.
  std::byte *writeRaw(std::byte *dst, const void *src, size_t size) {
    std::memcpy(dst, src, size);
    return dst + size;
  }

  // Helper wrapper around writeRaw for writing proper C++ types.
  template <typename T> std::byte *write(std::byte *dst, const T &src) {
    static_assert(std::is_trivially_copyable_v<T>,
                  "T must be trivially copyable");
    return writeRaw(dst, &src, sizeof(src));
  }

  // Executable Instance packaged up for serialization in our custom format.
  // As per the comments in SerializationHeader.
  std::vector<std::byte> m_payload;
};

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_SERIALIZED_EXECUTABLE_INSTANCE_H_
