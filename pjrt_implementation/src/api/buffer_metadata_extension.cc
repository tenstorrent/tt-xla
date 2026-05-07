// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/buffer_metadata_extension.h"

#include <optional>
#include <string_view>

#include "api/client_instance.h"
#include "utils/logging.h"

namespace tt::pjrt {

namespace {

// Log metadata entries for debugging
void logMetadata(const PJRT_BufferMetadata* metadata) {
  if (!metadata) {
    return;
  }

  LOG_F(INFO, "BufferMetadata: %zu entries", metadata->num_entries);

  for (size_t i = 0; i < metadata->num_entries; ++i) {
    const auto& entry = metadata->entries[i];
    std::string_view key(entry.key, entry.key_size);

    if (entry.value) {
      std::string_view value(entry.value, entry.value_size);
      LOG_F(INFO, "  [%zu] %.*s = \"%.*s\"",
             i,
             static_cast<int>(key.size()), key.data(),
             static_cast<int>(value.size()), value.data());
    } else {
      LOG_F(INFO, "  [%zu] %.*s = %ld",
             i,
             static_cast<int>(key.size()), key.data(),
             entry.int_value);
    }
  }
}

// Extract logical_id from metadata entries if present.
// Returns std::nullopt if not found.
std::optional<std::int64_t>
extractLogicalId(const PJRT_BufferMetadata* metadata) {
  if (!metadata) {
    return std::nullopt;
  }

  for (size_t i = 0; i < metadata->num_entries; ++i) {
    const auto& entry = metadata->entries[i];
    std::string_view key(entry.key, entry.key_size);

    // If key is "logical_id" and value is null, use int_value
    if (key == "logical_id" && entry.value == nullptr) {
      return entry.int_value;
    }
  }

  return std::nullopt;
}

PJRT_Error* onBufferFromHostBufferWithMetadata(
    PJRT_Client_BufferFromHostBufferWithMetadata_Args* args) {

  LOG_F(INFO, "BufferMetadataExtension::BufferFromHostBufferWithMetadata");

  // Log metadata if present
  logMetadata(args->metadata);

  // Extract logical_id from metadata for deferred sharding
  std::optional<std::int64_t> logical_id = extractLogicalId(args->metadata);
  if (logical_id.has_value()) {
    LOG_F(INFO, "Extracted logical_id=%lld from metadata",
          static_cast<long long>(logical_id.value()));
  }

  // Convert to standard BufferFromHostBuffer args and delegate to existing
  // implementation
  PJRT_Client_BufferFromHostBuffer_Args standard_args;
  standard_args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  standard_args.extension_start = args->extension_start;
  standard_args.client = args->client;
  standard_args.data = args->data;
  standard_args.type = args->type;
  standard_args.dims = args->dims;
  standard_args.num_dims = args->num_dims;
  standard_args.byte_strides = args->byte_strides;
  standard_args.num_byte_strides = args->num_byte_strides;
  standard_args.host_buffer_semantics = args->host_buffer_semantics;
  standard_args.device = args->device;
  standard_args.memory = args->memory;
  standard_args.device_layout = args->device_layout;

  // Call implementation with logical_id for deferred sharding support
  PJRT_Error* error =
      internal::onBufferFromHostBuffer(&standard_args, logical_id);

  // Copy outputs back
  args->done_with_host_buffer = standard_args.done_with_host_buffer;
  args->buffer = standard_args.buffer;

  return error;
}

} // anonymous namespace

// Static extension instance
static PJRT_BufferMetadata_Extension buffer_metadata_extension = {
    .base = {
        .struct_size = sizeof(PJRT_Extension_Base),
        .type = PJRT_Extension_Type_BufferMetadata,
        .next = nullptr,
    },
    .buffer_from_host_buffer_with_metadata = &onBufferFromHostBufferWithMetadata,
};

PJRT_Extension_Base* getBufferMetadataExtension() {
  return &buffer_metadata_extension.base;
}

} // namespace tt::pjrt
