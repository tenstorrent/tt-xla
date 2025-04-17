// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/pjrt_implementation/buffer_instance.h"

// c++ standard library includes
#include <stdexcept>
#include <thread>

// tt-mlir includes
#include "tt/runtime/utils.h"

// tt-xla includes
#include "common/pjrt_implementation/data_type_utils.h"
#include "common/pjrt_implementation/device_instance.h"
#include "common/pjrt_implementation/error_instance.h"
#include "common/status.h"

namespace tt::pjrt {

std::unique_ptr<BufferInstance> BufferInstance::createInputBufferInstance(
    PJRT_Buffer_Type data_type, const std::int64_t *dims, size_t num_dims,
    DeviceInstance *device) {
  struct make_unique_enabler : public BufferInstance {
    make_unique_enabler(PJRT_Buffer_Type data_type, const std::int64_t *dims,
                        size_t num_dims, DeviceInstance *device)
        : BufferInstance(data_type, dims, num_dims, device) {}
  };

  return std::make_unique<make_unique_enabler>(data_type, dims, num_dims,
                                               device);
}

std::unique_ptr<BufferInstance> BufferInstance::createOutputBufferInstance(
    const tt::runtime::Tensor &tensor, std::vector<std::uint32_t> &&dimensions,
    DeviceInstance *device) {
  struct make_unique_enabler : public BufferInstance {
    make_unique_enabler(const tt::runtime::Tensor &tensor,
                        std::vector<std::uint32_t> &&dimensions,
                        DeviceInstance *device)
        : BufferInstance(tensor, std::move(dimensions), device) {}
  };

  return std::make_unique<make_unique_enabler>(tensor, std::move(dimensions),
                                               device);
}

BufferInstance::BufferInstance(PJRT_Buffer_Type data_type,
                               const std::int64_t *dims, size_t num_dims,
                               DeviceInstance *device)
    : m_data_type(data_type), m_dimensions(dims, dims + num_dims),
      m_device(device),
      m_runtime_tensor(nullptr, nullptr, tt::runtime::DeviceRuntime::TTNN),
      m_data_ready(false), m_data_ready_event(nullptr),
      m_done_with_host_buffer_event(nullptr), m_data_deleted(false) {}

BufferInstance::BufferInstance(const tt::runtime::Tensor &tensor,
                               const std::vector<std::uint32_t> &dimensions,
                               DeviceInstance *device)
    : m_data_type(tt::pjrt::data_type_utils::convertRuntimeToPJRTDataType(
          tt::runtime::getTensorDataType(tensor))),
      m_dimensions(dimensions.begin(), dimensions.end()), m_device(device),
      m_runtime_tensor(tensor), m_data_ready(false),
      m_data_ready_event(nullptr), m_done_with_host_buffer_event(nullptr),
      m_data_deleted(false) {
  // We want to be in control when buffers are deallocated, which happens during
  // buffer destruction or on delete/destroy API calls.
  tt::runtime::setTensorRetain(m_runtime_tensor, /*retain=*/true);
}

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
  api->PJRT_Buffer_IsOnCpu = internal::onBufferIsOnCpu;
  api->PJRT_Buffer_Device = internal::onBufferDevice;
  api->PJRT_Buffer_ReadyEvent = internal::onBufferReadyEvent;
}

size_t BufferInstance::getRuntimeTensorSize() const {
  std::uint32_t runtime_tensor_size =
      tt::runtime::getTensorVolume(m_runtime_tensor) *
      tt::runtime::getTensorElementSize(m_runtime_tensor);

  return static_cast<size_t>(runtime_tensor_size);
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

  tt::runtime::deallocateTensor(m_runtime_tensor, /*force=*/true);

  m_data_deleted = true;
  if (m_done_with_host_buffer_event) {
    m_done_with_host_buffer_event->markAsReady(tt_pjrt_status::kSuccess);

    // TODO(mrakita): Revert.
    // https://github.com/openxla/xla/issues/25172
    delete m_done_with_host_buffer_event;
  }
}

void BufferInstance::copyFromHost(
    const void *host_buffer, PJRT_Buffer_Type data_type,
    const std::int64_t *dims, size_t num_dims, const std::int64_t *byte_strides,
    size_t num_byte_strides, PJRT_HostBufferSemantics host_buffer_semantics,
    EventInstance **out_done_with_host_buffer_event) {
  ::tt::target::DataType runtime_data_type =
      tt::pjrt::data_type_utils::convertPJRTToRuntimeDataType(m_data_type);
  std::uint32_t element_size =
      tt::runtime::utils::dataTypeElementSize(runtime_data_type);
  std::vector<std::uint32_t> shape = calculateShape(dims, num_dims);
  std::vector<std::uint32_t> strides =
      calculateStrides(num_dims, byte_strides, num_byte_strides, element_size);

  std::unique_ptr<EventInstance> done_with_host_buffer_event =
      EventInstance::createInstance();

  // In case when input host buffer has a semantic `ImmutableOnlyDuringCall`
  // we are not allowed to alias it directly, so we have to create owned host
  // tensor which copies buffer data. In JAX this semantic is used only for
  // copying scalars and numpy arrays, so the copy shouldn't take long. We can
  // mark the event as ready since we don't need the original host buffer
  // anymore.
  if (host_buffer_semantics ==
      PJRT_HostBufferSemantics_kImmutableOnlyDuringCall) {
    m_runtime_tensor = tt::runtime::createOwnedHostTensor(
        host_buffer, shape, strides, element_size, runtime_data_type);

    // Memory is copied, we don't need host buffer anymore.
    done_with_host_buffer_event->markAsReady(tt_pjrt_status::kSuccess);
  }
  // Otherwise when input host buffer has other semantic we are allowed to alias
  // it, so we can create borrowed host which doesn't copy any data and instead
  // uses direct pointer to existing data. Since we are holding a pointer to the
  // original data we can't mark the event as ready yet, so we remember it and
  // mark it as ready once the buffer is destroyed.
  else {
    // TODO(mrakita): Metal doesn't have a read-only version of borrowed buffer
    // so we have to const cast here.
    // https://github.com/tenstorrent/tt-metal/issues/20622
    m_runtime_tensor = tt::runtime::createBorrowedHostTensor(
        const_cast<void *>(host_buffer), shape, strides, element_size,
        runtime_data_type);

    // Memory is aliased, we need to hold on to host buffer until this buffer is
    // deleted.
    m_done_with_host_buffer_event = done_with_host_buffer_event.get();

    // TODO(mrakita): This is a major hack that we currently have to do because
    // XLA PJRT client destroys event immediately after it sets callback on it.
    // https://github.com/openxla/xla/issues/25172
    m_done_with_host_buffer_event->setIndestructible();
  }

  // We want to be in control when input buffers are deallocated, which happens
  // during buffer destruction or on delete/destroy API calls.
  tt::runtime::setTensorRetain(m_runtime_tensor, /*retain=*/true);

  markAsDataReady();

  // Releasing the ownership to the PJRT API caller since the caller is
  // responsible for calling `PJRT_Event_Destroy` on the event.
  *out_done_with_host_buffer_event = done_with_host_buffer_event.release();
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
  // Making sure that the host buffer size is greater than or equal to the
  // runtime tensor size.
  size_t runtime_tensor_size = getRuntimeTensorSize();
  if (runtime_tensor_size > host_buffer_size) {
    DLOG_F(ERROR,
           "Tried to copy device buffer to the host buffer with smaller size "
           "than required (device buffer size: %zu, host buffer size: %zu)",
           runtime_tensor_size, host_buffer_size);
    out_copy_done_event = nullptr;
    return tt_pjrt_status::kFailedPrecondition;
  }

  std::unique_ptr<EventInstance> event = EventInstance::createInstance();

  std::thread(
      [](void *host_buffer, tt::runtime::Tensor runtime_tensor,
         EventInstance *event) {
        tt_pjrt_status copy_status = tt_pjrt_status::kSuccess;
        try {
          tt::runtime::memcpy(host_buffer, runtime_tensor);
        } catch (const std::runtime_error &error) {
          DLOG_F(ERROR, "Copy to host buffer failed with error: %s",
                 error.what());
          copy_status = tt_pjrt_status::kInternal;
        }
        event->markAsReady(copy_status);
      },
      host_buffer, m_runtime_tensor, event.get())
      .join();

  // Releasing the ownership to the PJRT API caller since the caller is
  // responsible for calling `PJRT_Event_Destroy` on the event.
  *out_copy_done_event = event.release();

  return tt_pjrt_status::kSuccess;
}

void BufferInstance::markAsDataReady() {
  assert(!m_data_ready);

  std::lock_guard<std::mutex> ready_lock(m_data_ready_mutex);

  m_data_ready = true;
  if (m_data_ready_event) {
    m_data_ready_event->markAsReady(tt_pjrt_status::kSuccess);
  }
}

tt_pjrt_status BufferInstance::createDataReadyEvent(EventInstance **out_event) {
  if (m_data_ready_event) {
    DLOG_F(ERROR, "Buffer marked as data ready multiple times");
    return tt_pjrt_status::kInternal;
  }

  std::lock_guard<std::mutex> ready_lock(m_data_ready_mutex);

  std::unique_ptr<EventInstance> data_ready_event =
      EventInstance::createInstance();
  if (m_data_ready) {
    data_ready_event->markAsReady(tt_pjrt_status::kSuccess);
  }
  m_data_ready_event = data_ready_event.get();

  // Releasing the ownership to the PJRT API caller since the caller is
  // responsible for calling `PJRT_Event_Destroy` on the event.
  *out_event = data_ready_event.release();

  return tt_pjrt_status::kSuccess;
}

namespace internal {

PJRT_Error *onBufferDestroy(PJRT_Buffer_Destroy_Args *args) {
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Destroy");

  delete BufferInstance::unwrap(args->buffer);

  return nullptr;
}

PJRT_Error *onBufferElementType(PJRT_Buffer_ElementType_Args *args) {
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_ElementType");

  args->type = BufferInstance::unwrap(args->buffer)->getDataType();

  return nullptr;
}

PJRT_Error *onBufferDimensions(PJRT_Buffer_Dimensions_Args *args) {
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Dimensions");

  BufferInstance *buffer = BufferInstance::unwrap(args->buffer);
  args->dims = buffer->getDimensionsRaw();
  args->num_dims = buffer->getNumberOfDimensions();

  return nullptr;
}

PJRT_Error *
onBufferUnpaddedDimensions(PJRT_Buffer_UnpaddedDimensions_Args *args) {
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_UnpaddedDimensions");

  BufferInstance *buffer = BufferInstance::unwrap(args->buffer);
  // We don't support dynamic dimensions with padding yet.
  args->unpadded_dims = buffer->getDimensionsRaw();
  args->num_dims = buffer->getNumberOfDimensions();

  return nullptr;
}

PJRT_Error *onBufferDynamicDimensionIndices(
    PJRT_Buffer_DynamicDimensionIndices_Args *args) {
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_DynamicDimensionIndices");

  // We don't support dynamic dimensions yet.
  args->dynamic_dim_indices = nullptr;
  args->num_dynamic_dims = 0;

  return nullptr;
}

PJRT_Error *onBufferToHostBuffer(PJRT_Buffer_ToHostBuffer_Args *args) {
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
    args->dst_size = buffer->getRuntimeTensorSize();
    return nullptr;
  }

  return *ErrorInstance::makeError(
              buffer->copyToHost(
                  args->dst, args->dst_size,
                  reinterpret_cast<EventInstance **>(&args->event)))
              .release();
}

PJRT_Error *onBufferDelete(PJRT_Buffer_Delete_Args *args) {
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Delete");

  BufferInstance::unwrap(args->buffer)->deleteData();

  return nullptr;
}

PJRT_Error *onBufferIsDeleted(PJRT_Buffer_IsDeleted_Args *args) {
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_IsDeleted");

  args->is_deleted = BufferInstance::unwrap(args->buffer)->isDataDeleted();

  return nullptr;
}

PJRT_Error *onBufferIsOnCpu(PJRT_Buffer_IsOnCpu_Args *args) {
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_IsOnCpu");

  // Currently all our inputs are transferred to device where computation runs.
  args->is_on_cpu = false;

  return nullptr;
}

PJRT_Error *onBufferDevice(PJRT_Buffer_Device_Args *args) {
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Device");

  args->device = *BufferInstance::unwrap(args->buffer)->getDevice();

  return nullptr;
}

PJRT_Error *onBufferReadyEvent(PJRT_Buffer_ReadyEvent_Args *args) {
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_ReadyEvent");

  BufferInstance *buffer = BufferInstance::unwrap(args->buffer);

  return *ErrorInstance::makeError(
              buffer->createDataReadyEvent(
                  reinterpret_cast<EventInstance **>(&args->event)))
              .release();
}

} // namespace internal

} // namespace tt::pjrt
