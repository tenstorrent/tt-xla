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

// tt-xla includes
#include "common/pjrt_implementation/device_instance.h"
#include "common/pjrt_implementation/utils.h"

namespace tt::pjrt {

BufferInstance::BufferInstance(
    DeviceInstance &device, tt::runtime::Tensor &tensor,
    const std::vector<std::uint32_t> &shape,
    const std::vector<std::uint32_t> &stride,
    std::pair<tt::target::DataType, size_t> tt_buffer_type)
    : BufferInstance(device, tensor, shape, stride, tt_buffer_type, nullptr) {}

BufferInstance::BufferInstance(
    DeviceInstance &device, tt::runtime::Tensor &tensor,
    const std::vector<std::uint32_t> &shape,
    const std::vector<std::uint32_t> &stride,
    std::pair<tt::target::DataType, size_t> tt_buffer_type,
    std::shared_ptr<void> host_buffer_ptr)
    : device_(device), m_runtime_tensor(tensor),
      host_buffer_ptr_(host_buffer_ptr), tt_buffer_type_(tt_buffer_type),
      dims_(shape.begin(), shape.end()), stride_(stride) {}

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
  api->PJRT_Buffer_IsOnCpu =
      +[](PJRT_Buffer_IsOnCpu_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_IsOnCpu");
    args->is_on_cpu = BufferInstance::unwrap(args->buffer)->is_on_cpu();
    return nullptr;
  };
  api->PJRT_Buffer_Device = +[](PJRT_Buffer_Device_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Device");
    args->device = BufferInstance::unwrap(args->buffer)->device();
    return nullptr;
  };
  api->PJRT_Buffer_ReadyEvent = internal::onBufferReadyEvent;
  api->PJRT_Buffer_UnsafePointer =
      +[](PJRT_Buffer_UnsafePointer_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_UnsafePointer");
    return nullptr;
  };
}

size_t BufferInstance::getRuntimeTensorSize() const {
  std::uint32_t runtime_tensor_size =
      tt::runtime::getTensorVolume(m_runtime_tensor) *
      tt::runtime::getTensorElementSize(m_runtime_tensor);

  return static_cast<size_t>(runtime_tensor_size);
}

tt_pjrt_status BufferInstance::copyFromHost() {
  // TODO_OOM: finish
}

// TODO_OOM: remove
void *BufferInstance::getHostBuffer() {
  return m_host_buffer_copy ? m_host_buffer_copy.get() : m_aliased_host_buffer;
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
}

tt_pjrt_status BufferInstance::copyToHost(void *host_buffer,
                                          size_t host_buffer_size,
                                          EventInstance **out_event) {
  // Making sure that the host buffer size is greater than or equal to the
  // runtime tensor size.
  size_t runtime_tensor_size = getRuntimeTensorSize();
  if (runtime_tensor_size > host_buffer_size) {
    DLOG_F(ERROR,
           "Tried to copy device buffer to the host buffer with smaller size "
           "than required (device buffer size: %zu, host buffer size: %zu)",
           runtime_tensor_size, host_buffer_size);
    out_event = nullptr;
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
  // responsible for calling PJRT_Event_Destroy on event.
  *out_event = event.release();

  return tt_pjrt_status::kSuccess;
}

// TODO_OOM: remove
PJRT_Buffer_Type BufferInstance::getRuntimeType() {
  DLOG_F(LOG_DEBUG, "BufferInstance::element_type");
  tt::target::DataType Type = tt::runtime::getTensorDataType(getTensor());
  return tt::pjrt::utils::convertElementTypeToBufferType(Type);
}

namespace internal {

PJRT_Error *onBufferDestroy(PJRT_Buffer_Destroy_Args *args) {
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Destroy");

  BufferInstance *buffer = BufferInstance::unwrap(args->buffer);
  delete buffer;

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
  args->dims = buffer->getRawDimensions();
  args->num_dims = buffer->getNumberOfDimensions();

  return nullptr;
}

PJRT_Error *
onBufferUnpaddedDimensions(PJRT_Buffer_UnpaddedDimensions_Args *args) {
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_UnpaddedDimensions");

  BufferInstance *buffer = BufferInstance::unwrap(args->buffer);
  // We don't support dynamic dimensions with padding yet.
  args->unpadded_dims = buffer->getRawDimensions();
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

  // TODO_OOM: Check the args->host_layout. PJRT comment for that arg:
  // "The caller can specify an optional host layout. If nullptr, the layout of
  // the src buffer will be used. The caller is responsible to keep the data
  // (tiled or strides) in the host_layout alive during the call."
  if (args->host_layout) {
    DLOG_F(ERROR, "Copying to host with custom memory layout is not supported");
    return ErrorInstance::MakeError(tt_pjrt_status::kUnimplemented);
  }

  BufferInstance *buffer = BufferInstance::unwrap(args->src);

  // This API function can be used with null `dst` to query the required size.
  if (!args->dst) {
    args->dst_size = buffer->getRuntimeTensorSize();
    return nullptr;
  }

  return ErrorInstance::MakeError(
      buffer->copyToHost(args->dst, args->dst_size,
                         reinterpret_cast<EventInstance **>(&args->event)));
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

PJRT_Error *onBufferReadyEvent(PJRT_Buffer_ReadyEvent_Args *args) {
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_ReadyEvent");

  // TODO_OOM: finish
  BufferInstance *buffer = BufferInstance::unwrap(args->buffer);
  std::unique_ptr<EventInstance> onReadyEvent =
      std::make_unique<EventInstance>();
  buffer->on_ready_event_ = onReadyEvent.get();
  // Releasing the ownership to the PJRT API caller since the caller is
  // responsible for calling PJRT_Event_Destroy on event.
  args->event = *onReadyEvent.release();

  return nullptr;
}

} // namespace internal

} // namespace tt::pjrt
