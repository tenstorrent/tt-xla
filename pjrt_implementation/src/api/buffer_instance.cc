// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "api/buffer_instance.h"

// c++ standard library includes
#include <cstring>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <thread>

// POSIX includes
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// tracy includes
#include "tracy/Tracy.hpp"

// PJRT C API includes
#include "api/loaded_executable_instance.h"
#include "tt/runtime/types.h"
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-mlir includes
#include "tt/runtime/runtime.h"
#include "tt/runtime/utils.h"
#include "ttmlir/Target/Common/types_generated.h"

// tt-xla includes
#include "api/client_instance.h"
#include "api/device_instance.h"
#include "api/error_instance.h"
#include "api/memory_instance.h"
#include "utils/data_type_utils.h"
#include "utils/logging.h"
#include "utils/status.h"
#include "utils/utils.h"

namespace tt::pjrt {

// MmappedBuffer implementation
MmappedBuffer::MmappedBuffer(const void *source_data, size_t size)
    : m_data(nullptr), m_size(size) {
  DLOG_F(LOG_DEBUG, "MmappedBuffer: Creating mmap for %zu bytes (%.2f MB)",
         size, size / 1024.0 / 1024.0);

  // Create anonymous mmap with MAP_PRIVATE and MAP_ANONYMOUS
  // This allows the OS to swap pages to disk when memory pressure is high
  m_data = ::mmap(nullptr, size, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (m_data == MAP_FAILED) {
    LOG_F(ERROR, "MmappedBuffer: Failed to create mmap for %zu bytes: %s",
          size, std::strerror(errno));
    throw std::runtime_error("Failed to create mmap buffer: " +
                             std::string(std::strerror(errno)));
  }

  DLOG_F(LOG_DEBUG, "MmappedBuffer: Successfully created mmap at address %p, size %.2f MB",
         m_data, size / 1024.0 / 1024.0);

  // Copy data into the mmap'd region
  std::memcpy(m_data, source_data, size);

  DLOG_F(LOG_DEBUG, "MmappedBuffer: Copied %zu bytes into mmap'd region", size);

  // Advise the kernel that this memory can be swapped out aggressively
  // MADV_DONTNEED tells the kernel it can free these pages immediately if needed
  int madvise_result = ::madvise(m_data, size, MADV_DONTNEED);
  if (madvise_result == 0) {
    DLOG_F(LOG_DEBUG, "MmappedBuffer: madvise(MADV_DONTNEED) successful for %p", m_data);
  } else {
    DLOG_F(LOG_DEBUG, "MmappedBuffer: madvise(MADV_DONTNEED) failed for %p: %s",
           m_data, std::strerror(errno));
  }
}

MmappedBuffer::~MmappedBuffer() {
  if (m_data != nullptr && m_data != MAP_FAILED) {
    DLOG_F(LOG_DEBUG, "MmappedBuffer: Destroying mmap at address %p, size %.2f MB",
           m_data, m_size / 1024.0 / 1024.0);
    ::munmap(m_data, m_size);
  }
}

MmappedBuffer::MmappedBuffer(MmappedBuffer &&other) noexcept
    : m_data(other.m_data), m_size(other.m_size) {
  other.m_data = nullptr;
  other.m_size = 0;
}

MmappedBuffer &MmappedBuffer::operator=(MmappedBuffer &&other) noexcept {
  if (this != &other) {
    if (m_data != nullptr && m_data != MAP_FAILED) {
      ::munmap(m_data, m_size);
    }
    m_data = other.m_data;
    m_size = other.m_size;
    other.m_data = nullptr;
    other.m_size = 0;
  }
  return *this;
}

std::mutex BufferInstance::s_copy_to_host_internal_mutex;

std::unique_ptr<BufferInstance> BufferInstance::createInputBufferInstance(
    PJRT_Buffer_Type data_type, const std::int64_t *dims, size_t num_dims,
    DeviceInstance *device, MemoryInstance *memory) {
  struct make_unique_enabler : public BufferInstance {
    make_unique_enabler(PJRT_Buffer_Type data_type, const std::int64_t *dims,
                        size_t num_dims, DeviceInstance *device,
                        MemoryInstance *memory)
        : BufferInstance(data_type, dims, num_dims, device, memory) {}
  };

  return std::make_unique<make_unique_enabler>(data_type, dims, num_dims,
                                               device, memory);
}

std::unique_ptr<BufferInstance> BufferInstance::createOutputBufferInstance(
    std::vector<std::uint32_t> &&dimensions, DeviceInstance *device,
    MemoryInstance *memory, PJRT_Buffer_Type data_type, uint32_t device_id) {
  struct make_unique_enabler : public BufferInstance {
    make_unique_enabler(std::vector<std::uint32_t> &&dimensions,
                        DeviceInstance *device, MemoryInstance *memory,
                        PJRT_Buffer_Type data_type,
                        std::optional<uint32_t> device_id)
        : BufferInstance(std::move(dimensions), device, memory, data_type,
                         device_id) {}
  };

  return std::make_unique<make_unique_enabler>(std::move(dimensions), device,
                                               memory, data_type, device_id);
}

BufferInstance::BufferInstance(PJRT_Buffer_Type data_type,
                               const std::int64_t *dims, size_t num_dims,
                               DeviceInstance *device, MemoryInstance *memory)
    : m_uid(nextUID()), m_data_type(data_type),
      m_dimensions(dims, dims + num_dims), m_device(device),
      m_device_id(std::nullopt), m_memory(memory), m_data_ready(false),
      m_data_ready_event(nullptr), m_done_with_host_buffer_event(nullptr),
      m_data_deleted(false) {}

BufferInstance::BufferInstance(const std::vector<std::uint32_t> &dimensions,
                               DeviceInstance *device, MemoryInstance *memory,
                               PJRT_Buffer_Type data_type,
                               std::optional<uint32_t> device_id)
    : m_uid(nextUID()), m_data_type(data_type),
      m_dimensions(dimensions.begin(), dimensions.end()), m_device(device),
      m_device_id(device_id), m_memory(memory), m_data_ready(false),
      m_data_ready_event(nullptr), m_done_with_host_buffer_event(nullptr),
      m_data_deleted(false) {}

BufferInstance::~BufferInstance() { deleteData(); }

void BufferInstance::bindApi(PJRT_Api *api) {
  api->PJRT_Buffer_Destroy = internal::onBufferDestroy;
  api->PJRT_Buffer_ElementType = internal::onBufferElementType;
  api->PJRT_Buffer_Dimensions = internal::onBufferDimensions;
  api->PJRT_Buffer_UnpaddedDimensions = internal::onBufferUnpaddedDimensions;
  api->PJRT_Buffer_DynamicDimensionIndices =
      internal::onBufferDynamicDimensionIndices;
  api->PJRT_Buffer_ToHostBuffer = internal::onBufferToHostBuffer;
  api->PJRT_Buffer_Delete = internal::onBufferDelete;
  api->PJRT_Buffer_IsDeleted = internal::onBufferIsDeleted;
  api->PJRT_Buffer_CopyToDevice = internal::onBufferCopyToDevice;
  api->PJRT_Buffer_CopyToMemory = internal::onBufferCopyToMemory;
  api->PJRT_Buffer_IsOnCpu = internal::onBufferIsOnCpu;
  api->PJRT_Buffer_Device = internal::onBufferDevice;
  api->PJRT_Buffer_ReadyEvent = internal::onBufferReadyEvent;
  api->PJRT_Buffer_Memory = internal::onBufferMemory;
}

size_t BufferInstance::logicalTensorSize() const {
  size_t dtype_element_size = tt::runtime::utils::dataTypeElementSize(
      data_type_utils::convertPJRTToRuntimeDataType(m_data_type));

  return std::accumulate(m_dimensions.begin(), m_dimensions.end(),
                         dtype_element_size, [](size_t acc, std::int64_t dim) {
                           return acc * static_cast<size_t>(dim);
                         });
}

std::string BufferInstance::toShapeStr() const {
  return tt::pjrt::utils::to_string(m_dimensions);
}

bool BufferInstance::isDataDeleted() {
  std::lock_guard<std::mutex> deleted_lock(m_data_deleted_mutex);
  return m_data_deleted;
}

void BufferInstance::deleteData() {
  if (m_data_deleted) {
    return;
  }

  std::lock_guard<std::mutex> deleted_lock(m_data_deleted_mutex);
  if (m_data_deleted) {
    return;
  }

  joinCopyThread();

  // IMPORTANT: Destroy the borrowed tensor before the mmap'd buffer!
  // The borrowed tensor points to memory owned by m_mmapped_buffer,
  // so we must ensure the tensor is destroyed first.
  if (m_pjrt_tensor) {
    DLOG_F(LOG_DEBUG, "BufferInstance[UID=%lu]::deleteData - Destroying borrowed tensor", m_uid);
    m_pjrt_tensor.reset();
  }

  // Now it's safe to destroy the mmap'd buffer
  if (m_mmapped_buffer) {
    DLOG_F(LOG_DEBUG, "BufferInstance[UID=%lu]::deleteData - Destroying mmap at address %p, size %.2f MB",
           m_uid, m_mmapped_buffer->data(), m_mmapped_buffer->size() / 1024.0 / 1024.0);
    m_mmapped_buffer.reset();
  }

  m_data_deleted = true;
}

// Constructing buffer instance for the first time.
void BufferInstance::copyFromHost(
    const void *host_buffer, PJRT_Buffer_Type data_type,
    const std::int64_t *dims, size_t num_dims, const std::int64_t *byte_strides,
    size_t num_byte_strides, PJRT_HostBufferSemantics host_buffer_semantics,
    EventInstance **out_done_with_host_buffer_event) {

  assert(data_type == m_data_type && "m_data_type and data_type do not match");

  m_pjrt_tensor.reset();

  ::tt::target::DataType runtime_data_type =
      tt::pjrt::data_type_utils::convertPJRTToRuntimeDataType(m_data_type);
  std::uint32_t element_size =
      tt::runtime::utils::dataTypeElementSize(runtime_data_type);
  std::vector<std::uint32_t> shape = calculateShape(dims, num_dims);
  std::vector<std::uint32_t> strides =
      calculateStrides(num_dims, byte_strides, num_byte_strides, element_size);

  std::unique_ptr<EventInstance> done_with_host_buffer_event =
      EventInstance::createInstance();

  // Calculate the total size of the buffer
  size_t buffer_size = element_size;
  for (auto dim : shape) {
    buffer_size *= dim;
  }

  DLOG_F(LOG_DEBUG, "BufferInstance[UID=%lu]::copyFromHost - Creating mmap for buffer size %zu bytes (%.2f MB), shape=%s",
         m_uid, buffer_size, buffer_size / 1024.0 / 1024.0, toShapeStr().c_str());

  // Create an mmap'd buffer that can be offloaded to disk by the OS
  // This gives us memory pressure relief while maintaining data integrity
  m_mmapped_buffer = std::make_unique<MmappedBuffer>(host_buffer, buffer_size);

  DLOG_F(LOG_DEBUG, "BufferInstance[UID=%lu]::copyFromHost - Mmap created at address %p for %.2f MB",
         m_uid, m_mmapped_buffer->data(), buffer_size / 1024.0 / 1024.0);

  // Create a borrowed host tensor that points to the mmap'd memory
  // The borrowed tensor doesn't own the memory, so it won't free it on destruction
  // The mmap'd buffer will be cleaned up when this BufferInstance is destroyed
  tt::runtime::Tensor runtime_tensor = tt::runtime::createBorrowedHostTensor(
      m_mmapped_buffer->data(), shape, strides, element_size, runtime_data_type);

  DLOG_F(LOG_DEBUG, "BufferInstance[UID=%lu]::copyFromHost - Created borrowed tensor pointing to mmap at %p",
         m_uid, m_mmapped_buffer->data());

  // Memory is copied into mmap'd region, we don't need the original host buffer anymore.
  // Mark the event as ready immediately after the copy completes.
  EventInstance::markAsReadyAndCallback(done_with_host_buffer_event.get(),
                                        tt_pjrt_status::kSuccess);

  PjrtTensor::from_runtime_tensor({this}, runtime_tensor);

  markAsDataReady();

  // Releasing the ownership to the PJRT API caller since the caller is
  // responsible for calling `PJRT_Event_Destroy` on the event.
  *out_done_with_host_buffer_event = done_with_host_buffer_event.release();
}

void BufferInstance::copyFromBuffer(BufferInstance *src_buffer) {
  DLOG_F(LOG_DEBUG, "BufferInstance::copyFromBuffer");
  assert(src_buffer->getPjrtTensor() && "Source buffer has no data.");

  ::tt::target::DataType runtime_data_type =
      tt::pjrt::data_type_utils::convertPJRTToRuntimeDataType(
          src_buffer->m_data_type);

  std::uint32_t element_size =
      tt::runtime::utils::dataTypeElementSize(runtime_data_type);
  std::vector<std::uint32_t> shape = calculateShape(
      src_buffer->getDimensionsRaw(), src_buffer->getNumberOfDimensions());
  std::vector<std::uint32_t> strides = calculateStrides(
      src_buffer->getNumberOfDimensions(), nullptr, 0, element_size);

  tt::runtime::Tensor runtime_tensor = tt::runtime::createOwnedHostTensor(
      /* data= */ nullptr, shape, strides, element_size, runtime_data_type);

  src_buffer->getPjrtTensor()->move_to_host();
  tt::runtime::memcpy(runtime_tensor, src_buffer->runtimeTensor());

  PjrtTensor::from_runtime_tensor({this}, std::move(runtime_tensor));

  markAsDataReady();
}

std::vector<std::uint32_t>
BufferInstance::calculateShape(const std::int64_t *dims, size_t num_dims) {
  if (num_dims == 0) {
    // Our compiler and runtime don't support scalars so we convert them to 1D
    // tensors.
    return {1};
  }

  std::vector<std::uint32_t> shape;
  for (size_t i = 0; i < num_dims; ++i) {
    shape.push_back(dims[i]);
  }

  return shape;
}

std::vector<std::uint32_t> BufferInstance::calculateStrides(
    size_t num_dims, const std::int64_t *byte_strides, size_t num_byte_strides,
    std::uint32_t element_size) {
  if (num_dims == 0) {
    // Our compiler and runtime don't support scalars so we convert them to 1D
    // tensors.
    return {1};
  }

  assert(num_byte_strides == 0 || num_byte_strides == num_dims);

  std::vector<std::uint32_t> strides;
  for (size_t i = 0; i < num_dims; ++i) {
    // If no strides are given the array is assumed to have a dense layout with
    // dimensions in major-to-minor order.
    std::uint32_t stride =
        num_byte_strides == 0
            ? 1
            : (byte_strides[i] / static_cast<std::int64_t>(element_size));
    strides.push_back(stride);
  }

  return strides;
}

tt_pjrt_status BufferInstance::copyToHost(void *host_buffer,
                                          size_t host_buffer_size,
                                          EventInstance **out_copy_done_event) {
  ZoneScoped;
  assert(m_pjrt_tensor && "Copy from buffer without an associated tensor.");

  auto rt_data_type =
      tt::pjrt::data_type_utils::convertPJRTToRuntimeDataType(m_data_type);

  // TODO(acolic): Copying in a separate thread is left to match previous
  // behavior. Check whether it is needed: it does not make much sense to
  // create new thread for each shard, because tensors are moved once from
  // device when onHost is called on a first shard; on the other hand, there is
  // no sense to create new thread for memcpy because framework would wait on a
  // copy anyway, right? Also, creating std::thread for memcpy might be overhead
  // with performance loss. We must measure.
  // Also, std::thread (as all objects from std::) has a value semantic, so it
  // does not make any sense to create std::thread as a unique_ptr.
  joinCopyThread();

  std::unique_ptr<EventInstance> event = EventInstance::createInstance();

  m_copy_to_host_thread = std::make_unique<std::thread>([=, e = event.get()] {
    try {
      ZoneScopedN("CopyToHostThread");
      const std::lock_guard<std::mutex> lock(s_copy_to_host_internal_mutex);

      m_pjrt_tensor->move_to_host();

      assert(logicalTensorSize() <= host_buffer_size &&
             "Host buffer is too small.");
      tt::runtime::memcpy(host_buffer, m_pjrt_tensor->runtime_tensor(),
                          rt_data_type);

      EventInstance::markAsReadyAndCallback(e, tt_pjrt_status::kSuccess);

    } catch (const std::exception &error) {
      LOG_F(ERROR, "Copy to host buffer failed with error: %s", error.what());
      EventInstance::markAsReadyAndCallback(e, tt_pjrt_status::kInternal);
    }
  });

  // responsible for calling `PJRT_Event_Destroy` on the event.
  *out_copy_done_event = event.release();

  return tt_pjrt_status::kSuccess;
}

void BufferInstance::markAsDataReady() {
  assert(!m_data_ready);

  std::lock_guard<std::mutex> ready_lock(m_data_ready_mutex);

  m_data_ready = true;
  if (m_data_ready_event) {
    EventInstance::markAsReadyAndCallback(m_data_ready_event,
                                          tt_pjrt_status::kSuccess);
  }
}

// We do not copy to device memory. We are just coping buffer from provided pjrt
// buffer and pretend that we have copied buffer instance to device.
tt_pjrt_status BufferInstance::copyToDeviceMemory(DeviceInstance *dst_device,
                                                  MemoryInstance *dst_memory,
                                                  BufferInstance **dst_buffer) {
  // PJRT API specification requires returning error in case of copying to same
  // device/memory space.
  if (getMemory() == dst_memory || getDevice() == dst_device) {
    LOG_F(ERROR, "Cannot copy buffer to the same memory or device");
    return tt_pjrt_status::kInvalidArgument;
  }

  std::unique_ptr<BufferInstance> dst_buffer_instance =
      BufferInstance::createInputBufferInstance(
          getDataType(), getDimensionsRaw(), getNumberOfDimensions(),
          dst_device, dst_memory);

  dst_buffer_instance->copyFromBuffer(this);

  *dst_buffer = dst_buffer_instance.release();

  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status BufferInstance::createDataReadyEvent(EventInstance **out_event) {
  if (m_data_ready_event) {
    LOG_F(ERROR, "Buffer marked as data ready multiple times");
    return tt_pjrt_status::kInternal;
  }

  std::lock_guard<std::mutex> ready_lock(m_data_ready_mutex);

  std::unique_ptr<EventInstance> data_ready_event =
      EventInstance::createInstance();
  if (m_data_ready) {
    EventInstance::markAsReadyAndCallback(data_ready_event.get(),
                                          tt_pjrt_status::kSuccess);
  }
  m_data_ready_event = data_ready_event.get();

  // Releasing the ownership to the PJRT API caller since the caller is
  // responsible for calling `PJRT_Event_Destroy` on the event.
  *out_event = data_ready_event.release();

  return tt_pjrt_status::kSuccess;
}

namespace internal {

PJRT_Error *onBufferDestroy(PJRT_Buffer_Destroy_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Destroy");

  delete BufferInstance::unwrap(args->buffer);

  return nullptr;
}

PJRT_Error *onBufferElementType(PJRT_Buffer_ElementType_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_ElementType");

  args->type = BufferInstance::unwrap(args->buffer)->getDataType();

  return nullptr;
}

PJRT_Error *onBufferDimensions(PJRT_Buffer_Dimensions_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Dimensions");

  BufferInstance *buffer = BufferInstance::unwrap(args->buffer);
  args->dims = buffer->getDimensionsRaw();
  args->num_dims = buffer->getNumberOfDimensions();

  return nullptr;
}

PJRT_Error *
onBufferUnpaddedDimensions(PJRT_Buffer_UnpaddedDimensions_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_UnpaddedDimensions");

  BufferInstance *buffer = BufferInstance::unwrap(args->buffer);
  // We don't support dynamic dimensions with padding yet.
  args->unpadded_dims = buffer->getDimensionsRaw();
  args->num_dims = buffer->getNumberOfDimensions();

  return nullptr;
}

PJRT_Error *onBufferDynamicDimensionIndices(
    PJRT_Buffer_DynamicDimensionIndices_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_DynamicDimensionIndices");

  // We don't support dynamic dimensions yet.
  args->dynamic_dim_indices = nullptr;
  args->num_dynamic_dims = 0;

  return nullptr;
}

PJRT_Error *onBufferToHostBuffer(PJRT_Buffer_ToHostBuffer_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_ToHostBuffer");

  // TODO(mrakita): The caller can specify an optional `host_layout` arg to
  // specify the memory layout in which data should be copied. It can sometimes
  // be tiled layout but we support only strided, which might explain accuracy
  // issues for some models. We need to investigate and add support for both
  // layouts.
  // https://github.com/tenstorrent/tt-xla/issues/500

  BufferInstance *buffer = BufferInstance::unwrap(args->src);

  // This API function can be used with null `dst` to query the required size.
  if (!args->dst) {
    ZoneScopedN("QueryHostBufferSize");
    DLOG_F(LOG_DEBUG, "Querying host buffer size");

    args->dst_size = buffer->logicalTensorSize();

    return nullptr;
  }

  return *ErrorInstance::makeError(
              buffer->copyToHost(
                  args->dst, args->dst_size,
                  reinterpret_cast<EventInstance **>(&args->event)))
              .release();
}

PJRT_Error *onBufferDelete(PJRT_Buffer_Delete_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Delete");

  BufferInstance::unwrap(args->buffer)->deleteData();

  return nullptr;
}

PJRT_Error *onBufferIsDeleted(PJRT_Buffer_IsDeleted_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_IsDeleted");

  args->is_deleted = BufferInstance::unwrap(args->buffer)->isDataDeleted();

  return nullptr;
}

// We do not copy to device memory. We are just coping buffer from provided pjrt
// buffer and pretend that we have copied buffer instance to device.
PJRT_Error *onBufferCopyToDevice(PJRT_Buffer_CopyToDevice_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_CopyToDevice");

  BufferInstance *src_buffer = BufferInstance::unwrap(args->buffer);
  DeviceInstance *dst_device = DeviceInstance::unwrap(args->dst_device);
  MemoryInstance *dst_memory = dst_device->getDefaultMemory();

  src_buffer->copyToDeviceMemory(
      dst_device, dst_memory,
      reinterpret_cast<BufferInstance **>(&args->dst_buffer));

  return nullptr;
}

PJRT_Error *onBufferCopyToMemory(PJRT_Buffer_CopyToMemory_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_CopyToMemory");

  BufferInstance *src_buffer = BufferInstance::unwrap(args->buffer);
  MemoryInstance *dst_memory = MemoryInstance::unwrap(args->dst_memory);
  DeviceInstance *dst_device = dst_memory->getDevice();

  // Copying into to host memory is undefined, since it's not clear
  // when we should use it, since PJRT_Buffer_ToHostBuffer is used for this.
  if (dst_memory->isHostMemory()) {
    LOG_F(ERROR, "Copying buffer to host memory is not supported");
    return *ErrorInstance::makeError(tt_pjrt_status::kUnimplemented).release();
  }

  src_buffer->copyToDeviceMemory(
      dst_device, dst_memory,
      reinterpret_cast<BufferInstance **>(&args->dst_buffer));

  return nullptr;
}

PJRT_Error *onBufferIsOnCpu(PJRT_Buffer_IsOnCpu_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_IsOnCpu");

  // Currently all our inputs are transferred to device where computation runs.
  args->is_on_cpu = false;

  return nullptr;
}

PJRT_Error *onBufferDevice(PJRT_Buffer_Device_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Device");

  args->device = *BufferInstance::unwrap(args->buffer)->getDevice();

  return nullptr;
}

PJRT_Error *onBufferMemory(PJRT_Buffer_Memory_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Memory");

  args->memory = *BufferInstance::unwrap(args->buffer)->getMemory();

  return nullptr;
}

PJRT_Error *onBufferReadyEvent(PJRT_Buffer_ReadyEvent_Args *args) {
  ZoneScoped;
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_ReadyEvent");

  BufferInstance *buffer = BufferInstance::unwrap(args->buffer);

  return *ErrorInstance::makeError(
              buffer->createDataReadyEvent(
                  reinterpret_cast<EventInstance **>(&args->event)))
              .release();
}

} // namespace internal

} // namespace tt::pjrt
