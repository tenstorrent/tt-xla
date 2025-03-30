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
int BufferInstance::id_counter_ = 0;

BufferInstance::~BufferInstance() = default;

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
      dims_(shape.begin(), shape.end()), stride_(stride) {
  DLOG_F(LOG_DEBUG, "BufferInstance::BufferInstance");
  unique_id_ = id_counter_++;
}

void BufferInstance::ComputeLayout() {
  DLOG_F(LOG_DEBUG, "BufferInstance::ComputeLayout");
}

void BufferInstance::BindApi(PJRT_Api *api) {
  DLOG_F(LOG_DEBUG, "BufferInstance::BindApi");
  api->PJRT_Buffer_Destroy =
      +[](PJRT_Buffer_Destroy_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Destroy");
    BufferInstance *buffer = BufferInstance::Unwrap(args->buffer);
    delete buffer;
    return nullptr;
  };
  api->PJRT_Buffer_ElementType =
      +[](PJRT_Buffer_ElementType_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_ElementType");
    BufferInstance *buffer = BufferInstance::Unwrap(args->buffer);
    std::optional<PJRT_Buffer_Type> type = buffer->getType();
    if (type.has_value())
      args->type = *type;
    else {
      args->type = buffer->getRuntimeType();
    }
    return nullptr;
  };
  api->PJRT_Buffer_Dimensions =
      +[](PJRT_Buffer_Dimensions_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Dimensions");
    BufferInstance *buffer = BufferInstance::Unwrap(args->buffer);
    args->dims = buffer->getRawDimensions();
    args->num_dims = buffer->num_dims();
    return nullptr;
  };
  api->PJRT_Buffer_UnpaddedDimensions =
      +[](PJRT_Buffer_UnpaddedDimensions_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_UnpaddedDimensions");
    BufferInstance *buffer = BufferInstance::Unwrap(args->buffer);
    args->unpadded_dims = buffer->getRawDimensions();
    args->num_dims = buffer->num_dims();
    return nullptr;
  };
  api->PJRT_Buffer_ToHostBuffer = internal::onBufferToHostBuffer;
  api->PJRT_Buffer_OnDeviceSizeInBytes =
      +[](PJRT_Buffer_OnDeviceSizeInBytes_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_OnDeviceSizeInBytes");
    BufferInstance *buffer = BufferInstance::Unwrap(args->buffer);
    return nullptr;
  };
  api->PJRT_Buffer_Delete = +[](PJRT_Buffer_Delete_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Delete");
    BufferInstance *buffer = BufferInstance::Unwrap(args->buffer);
    buffer->Delete();
    return nullptr;
  };
  api->PJRT_Buffer_IsDeleted =
      +[](PJRT_Buffer_IsDeleted_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_IsDeleted");
    BufferInstance *buffer = BufferInstance::Unwrap(args->buffer);
    args->is_deleted = buffer->is_deleted();
    return nullptr;
  };
  api->PJRT_Buffer_IsOnCpu =
      +[](PJRT_Buffer_IsOnCpu_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_IsOnCpu");
    args->is_on_cpu = BufferInstance::Unwrap(args->buffer)->is_on_cpu();
    return nullptr;
  };
  api->PJRT_Buffer_Device = +[](PJRT_Buffer_Device_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Device");
    args->device = BufferInstance::Unwrap(args->buffer)->device();
    return nullptr;
  };
  api->PJRT_Buffer_ReadyEvent = internal::onBufferReadyEvent;
  // TODO: Rework the API to be Aliases(b1, b2) to let the plugin explicitly
  // check for aliases.
  api->PJRT_Buffer_GetMemoryLayout =
      +[](PJRT_Buffer_GetMemoryLayout_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_GetMemoryLayout");
    BufferInstance *buffer = BufferInstance::Unwrap(args->buffer);
    return buffer->GetMemoryLayout(args);
  };
  api->PJRT_Buffer_UnsafePointer =
      +[](PJRT_Buffer_UnsafePointer_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_UnsafePointer");
    return nullptr;
  };
}

PJRT_Error *
BufferInstance::GetMemoryLayout(PJRT_Buffer_GetMemoryLayout_Args *args) {
  DLOG_F(LOG_DEBUG, "BufferInstance::GetMemoryLayout");
  args->layout.type =
      PJRT_Buffer_MemoryLayout_Type::PJRT_Buffer_MemoryLayout_Type_Tiled;
  size_t rank = num_dims();
  minor_to_major_.resize(rank);
  for (size_t i = 0; i < rank; i++) {
    minor_to_major_[i] = rank - 1 - i;
  }
  tile_dim_sizes_.resize(1);
  tile_dim_sizes_[0] = rank;
  tile_dims_.resize(rank);
  for (size_t i = 0; i < rank; i++) {
    tile_dims_[i] = dims_[i];
  }
  args->layout.tiled.minor_to_major_size = rank;
  args->layout.tiled.minor_to_major = minor_to_major_.data();
  args->layout.tiled.tile_dims = tile_dims_.data();
  args->layout.tiled.tile_dim_sizes = tile_dim_sizes_.data();

  args->layout.tiled.num_tiles = 1;
  return nullptr;
}

tt_pjrt_status BufferInstance::GetHostSizeInBytes(size_t *host_size) {
  DLOG_F(LOG_DEBUG, "BufferInstance::GetHostSizeInBytes");
  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status BufferInstance::AsyncDeallocate() {
  DLOG_F(LOG_DEBUG, "BufferInstance::AsyncDeallocate");
  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status BufferInstance::Delete() {
  DLOG_F(LOG_DEBUG, "BufferInstance::Delete");
  is_deleted_ = true;
  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status BufferInstance::copyToHost(void *host_buffer,
                                          size_t host_buffer_size,
                                          EventInstance **out_event) {
  // Making sure that the host buffer size is greater than or equal to the
  // runtime tensor size.
  std::uint32_t runtime_tensor_size =
      tt::runtime::getTensorVolume(m_runtime_tensor) *
      tt::runtime::getTensorElementSize(m_runtime_tensor);
  if (static_cast<size_t>(runtime_tensor_size) > host_buffer_size) {
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

PJRT_Buffer_Type BufferInstance::getRuntimeType() {
  DLOG_F(LOG_DEBUG, "BufferInstance::element_type");
  tt::target::DataType Type = tt::runtime::getTensorDataType(getTensor());
  return tt::pjrt::utils::convertElementTypeToBufferType(Type);
}

namespace internal {

PJRT_Error *onBufferToHostBuffer(PJRT_Buffer_ToHostBuffer_Args *args) {
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_ToHostBuffer");

  BufferInstance *buffer = BufferInstance::Unwrap(args->src);
  if (args->dst) {
    // Initiate transfer.
    return ErrorInstance::MakeError(
        buffer->copyToHost(args->dst, args->dst_size,
                           reinterpret_cast<EventInstance **>(&args->event)));
  } else {
    // Size query.
    return ErrorInstance::MakeError(
        buffer->GetHostSizeInBytes(&args->dst_size));
  }
}

PJRT_Error *onBufferReadyEvent(PJRT_Buffer_ReadyEvent_Args *args) {
  DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_ReadyEvent");

  BufferInstance *buffer = BufferInstance::Unwrap(args->buffer);
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
