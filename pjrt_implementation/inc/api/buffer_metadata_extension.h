// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2024 The PyTorch/XLA Authors
// SPDX-License-Identifier: Apache-2.0

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_BUFFER_METADATA_EXTENSION_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_BUFFER_METADATA_EXTENSION_H_

#include <stddef.h>
#include <stdint.h>

#include "xla/pjrt/c/pjrt_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Extension version for ABI compatibility checks.
#define PJRT_API_BUFFER_METADATA_EXTENSION_VERSION 1

// Custom extension type for buffer metadata.
// Uses a high value (1000) to avoid conflicts with upstream PJRT extension
// types. PJRT plugins that support this extension should check for this type
// when walking the extension_start linked list.
#define PJRT_Extension_Type_BufferMetadata ((PJRT_Extension_Type)1000)

// A single key-value metadata entry.
// Either value (string) or int_value can be used, depending on the key's
// semantics. If value is non-null, it's a string value; otherwise use
// int_value.
typedef struct PJRT_BufferMetadata_Entry {
  size_t struct_size;
  // The key name (not null-terminated, use key_size).
  const char* key;
  size_t key_size;
  // String value (may be null if using int_value instead).
  const char* value;
  size_t value_size;
  // Integer value (used when value is null).
  int64_t int_value;
} PJRT_BufferMetadata_Entry;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_BufferMetadata_Entry, int_value);

// A collection of metadata entries associated with a buffer transfer.
typedef struct PJRT_BufferMetadata {
  size_t struct_size;
  // Array of metadata entries.
  PJRT_BufferMetadata_Entry* entries;
  size_t num_entries;
} PJRT_BufferMetadata;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_BufferMetadata, num_entries);

// Arguments for BufferFromHostBufferWithMetadata.
// Mirrors PJRT_Client_BufferFromHostBuffer_Args with an additional metadata
// field.
typedef struct PJRT_Client_BufferFromHostBufferWithMetadata_Args {
  size_t struct_size;
  PJRT_Extension_Base* extension_start;

  // Standard BufferFromHostBuffer fields
  PJRT_Client* client;
  const void* data;
  PJRT_Buffer_Type type;
  const int64_t* dims;
  size_t num_dims;
  const int64_t* byte_strides;
  size_t num_byte_strides;
  PJRT_HostBufferSemantics host_buffer_semantics;
  PJRT_Device* device;
  PJRT_Memory* memory;
  PJRT_Buffer_MemoryLayout* device_layout;

  // Metadata associated with this buffer transfer.
  // May be null if no metadata is provided.
  const PJRT_BufferMetadata* metadata;

  // Output: Event indicating when it's safe to free `data`.
  // The caller is responsible for calling PJRT_Event_Destroy.
  PJRT_Event* done_with_host_buffer;  // out

  // Output: The created device buffer.
  // The caller is responsible for calling PJRT_Buffer_Destroy.
  PJRT_Buffer* buffer;  // out
} PJRT_Client_BufferFromHostBufferWithMetadata_Args;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_Client_BufferFromHostBufferWithMetadata_Args,
                          buffer);

// Function signature for creating a buffer with metadata.
typedef PJRT_Error* (*PJRT_Client_BufferFromHostBufferWithMetadata)(
    PJRT_Client_BufferFromHostBufferWithMetadata_Args* args);

// Extension structure that PJRT plugins can implement to support buffer
// metadata.
// Plugins should add this to their extension_start linked list with
// base.type = PJRT_Extension_Type_BufferMetadata.
typedef struct PJRT_BufferMetadata_Extension {
  PJRT_Extension_Base base;
  // Function pointer for creating buffers with metadata.
  // If this is null, the extension is present but not functional.
  PJRT_Client_BufferFromHostBufferWithMetadata
      buffer_from_host_buffer_with_metadata;
} PJRT_BufferMetadata_Extension;
PJRT_DEFINE_STRUCT_TRAITS(PJRT_BufferMetadata_Extension,
                          buffer_from_host_buffer_with_metadata);

#ifdef __cplusplus
}
#endif

// C++ API for tt-xla
#ifdef __cplusplus
namespace tt::pjrt {

// Returns the buffer metadata extension for registration in the PJRT API.
PJRT_Extension_Base* getBufferMetadataExtension();

} // namespace tt::pjrt
#endif

#endif  // TT_XLA_PJRT_IMPLEMENTATION_INC_API_BUFFER_METADATA_EXTENSION_H_
