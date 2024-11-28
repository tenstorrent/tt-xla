// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// This file incorporates work covered by the following copyright and permission
// notice:
// SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// https://llvm.org/LICENSE.txt

#include "common/api_impl.h"

// c++ standard library includes
#include <cassert>
#include <cstring>
#include <iostream>
#include <optional>
#include <sstream>
#include <utility>

// tt-xla includes
#include "common/module_builder.h"
#include "common/plugin_attributes.h"
#include "common/status.h"

namespace tt::pjrt {

std::pair<tt::target::DataType, size_t>
MapBufferTypeToElementType(PJRT_Buffer_Type buffer_type) {
  switch (buffer_type) {
  case PJRT_Buffer_Type_U8:
    return std::make_pair(tt::target::DataType::UInt8, 1);
  case PJRT_Buffer_Type_U16:
    return std::make_pair(tt::target::DataType::UInt16, 2);
  case PJRT_Buffer_Type_U32:
    return std::make_pair(tt::target::DataType::UInt32, 4);
  case PJRT_Buffer_Type_F16:
    return std::make_pair(tt::target::DataType::Float16, 2);
  case PJRT_Buffer_Type_F32:
    return std::make_pair(tt::target::DataType::Float32, 4);
  case PJRT_Buffer_Type_BF16:
    return std::make_pair(tt::target::DataType::BFloat16, 2);
  case PJRT_Buffer_Type_INVALID:
  case PJRT_Buffer_Type_S4:
  case PJRT_Buffer_Type_S8:
  case PJRT_Buffer_Type_S16:
  case PJRT_Buffer_Type_S32:
  case PJRT_Buffer_Type_S64:
  case PJRT_Buffer_Type_U4:
  case PJRT_Buffer_Type_PRED:
  case PJRT_Buffer_Type_U64:
  case PJRT_Buffer_Type_F64:
  case PJRT_Buffer_Type_C64:
  case PJRT_Buffer_Type_C128:
  default:
    assert(false && "Unsupported buffer type");
    return std::make_pair(tt::target::DataType::BFloat16, 2);
  }
}

static PJRT_Buffer_Type
convertElementTypeToBufferType(tt::target::DataType ElementType) {
  switch (ElementType) {
  case tt::target::DataType::UInt8:
    return PJRT_Buffer_Type_U8;
  case tt::target::DataType::UInt16:
    return PJRT_Buffer_Type_U16;
  case tt::target::DataType::UInt32:
    return PJRT_Buffer_Type_U32;
  case tt::target::DataType::Float16:
    return PJRT_Buffer_Type_F16;
  case tt::target::DataType::Float32:
    return PJRT_Buffer_Type_F32;
  case tt::target::DataType::BFloat16:
    return PJRT_Buffer_Type_BF16;
  default:
    assert(false && "Unsupported data type");
    return PJRT_Buffer_Type_BF16;
  }
}

const std::string_view kMlirFormat = "mlir";
//===----------------------------------------------------------------------===//
// Error
//===----------------------------------------------------------------------===//

void ErrorInstance::BindApi(PJRT_Api *api) {
  api->PJRT_Error_Destroy = +[](PJRT_Error_Destroy_Args *args) {
    if (!args->error)
      return;
    delete ErrorInstance::FromError(args->error);
  };
  api->PJRT_Error_Message = +[](PJRT_Error_Message_Args *args) {
    auto *error = ErrorInstance::FromError(args->error);
    if (!error) {
      args->message = "OK";
      args->message_size = 2;
      return;
    }

    const std::string &message = error->message();
    args->message = message.data();
    args->message_size = message.size();
  };
  api->PJRT_Error_GetCode = +[](PJRT_Error_GetCode_Args *args) -> PJRT_Error * {
    auto *error = ErrorInstance::FromError(args->error);
    tt_pjrt_status status = error->status();
    switch (status) {
    case tt_pjrt_status::kCancelled:
      args->code = PJRT_Error_Code_CANCELLED;
      break;
    case tt_pjrt_status::kUnknown:
      args->code = PJRT_Error_Code_UNKNOWN;
      break;
    case tt_pjrt_status::kInvalidArgument:
      args->code = PJRT_Error_Code_INVALID_ARGUMENT;
      break;
    case tt_pjrt_status::kDeadlineExceeded:
      args->code = PJRT_Error_Code_DEADLINE_EXCEEDED;
      break;
    case tt_pjrt_status::kNotFound:
      args->code = PJRT_Error_Code_NOT_FOUND;
      break;
    case tt_pjrt_status::kAlreadyExists:
      args->code = PJRT_Error_Code_ALREADY_EXISTS;
      break;
    case tt_pjrt_status::kPermissionDenied:
      args->code = PJRT_Error_Code_PERMISSION_DENIED;
      break;
    case tt_pjrt_status::kResourceExhausted:
      args->code = PJRT_Error_Code_RESOURCE_EXHAUSTED;
      break;
    case tt_pjrt_status::kFailedPrecondition:
      args->code = PJRT_Error_Code_FAILED_PRECONDITION;
      break;
    case tt_pjrt_status::kAborted:
      args->code = PJRT_Error_Code_ABORTED;
      break;
    case tt_pjrt_status::kOutOfRange:
      args->code = PJRT_Error_Code_OUT_OF_RANGE;
      break;
    case tt_pjrt_status::kUnimplemented:
      args->code = PJRT_Error_Code_UNIMPLEMENTED;
      break;
    case tt_pjrt_status::kInternal:
      args->code = PJRT_Error_Code_INTERNAL;
      break;
    case tt_pjrt_status::kUnavailable:
      args->code = PJRT_Error_Code_UNAVAILABLE;
      break;
    case tt_pjrt_status::kDataLoss:
      args->code = PJRT_Error_Code_DATA_LOSS;
      break;
    case tt_pjrt_status::kUnauthenticated:
      args->code = PJRT_Error_Code_UNAUTHENTICATED;
      break;
    default:
      // Should not happen.
      args->code = PJRT_Error_Code_UNKNOWN;
    }
    return nullptr;
  };
}

const std::string &ErrorInstance::message() const {
  std::string buffer;
  buffer.reserve(256);
  buffer += "Error code: ";
  buffer += std::to_string(static_cast<int>(status_));
  cached_message_ = std::move(buffer);
  return cached_message_;
}

//===----------------------------------------------------------------------===//
// BufferInstance
//===----------------------------------------------------------------------===//
int BufferInstance::id_counter_ = 0;

BufferInstance::~BufferInstance() = default;

BufferInstance::BufferInstance(DeviceInstance &device,
                               tt::runtime::Tensor tensor,
                               std::vector<std::uint32_t> shape,
                               std::vector<std::uint32_t> stride)
    : device_(device) {
  DLOG_F(LOG_DEBUG, "BufferInstance::BufferInstance");
  tensor_ = tensor;
  dims_.resize(shape.size());
  for (int i = 0; i < shape.size(); i++) {
    dims_[i] = shape[i];
  }
  stride_ = stride;
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
    args->dims = buffer->dims();
    args->num_dims = buffer->num_dims();
    return nullptr;
  };
  api->PJRT_Buffer_UnpaddedDimensions =
      +[](PJRT_Buffer_UnpaddedDimensions_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_UnpaddedDimensions");
    BufferInstance *buffer = BufferInstance::Unwrap(args->buffer);
    args->unpadded_dims = buffer->dims();
    args->num_dims = buffer->num_dims();
    return nullptr;
  };
  api->PJRT_Buffer_ToHostBuffer =
      +[](PJRT_Buffer_ToHostBuffer_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_ToHostBuffer");
    BufferInstance *buffer = BufferInstance::Unwrap(args->src);
    if (!args->dst) {
      // Size query.
      return MakeError(buffer->GetHostSizeInBytes(&args->dst_size));
    } else {
      // Initiate transfer.
      return MakeError(
          buffer->CopyToHost(args->dst, args->dst_size,
                             reinterpret_cast<EventInstance **>(&args->event)));
    }
  };
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
  api->PJRT_Buffer_CopyToDevice =
      +[](PJRT_Buffer_CopyToDevice_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_CopyToDevice");
    return MakeError(tt_pjrt_status::kUnimplemented);
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
  api->PJRT_Buffer_Memory = +[](PJRT_Buffer_Memory_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_Memory");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Buffer_ReadyEvent =
      +[](PJRT_Buffer_ReadyEvent_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_ReadyEvent");
    return nullptr;
  };
  // TODO: Rework the API to be Aliases(b1, b2) to let the plugin explicitly
  // check for aliases.
  api->PJRT_Buffer_GetMemoryLayout =
      +[](PJRT_Buffer_GetMemoryLayout_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "BufferInstance::PJRT_Buffer_GetMemoryLayout");
    auto *buffer = BufferInstance::Unwrap(args->buffer);
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
    tile_dims_[i] = dims()[i];
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

tt_pjrt_status BufferInstance::CopyToHost(void *dst, size_t dst_size,
                                          EventInstance **out_done_event) {
  DLOG_F(LOG_DEBUG, "BufferInstance::CopyToHost");

  // This callback simply deletes the `dst_buffer_ready_event`. We could perform
  // this deletion in the `dst_buffer_callback`, but this would result in the
  // callback thread of `dst_buffer_ready_event` detaching from the main thread,
  // potentially resulting in the callback thread outliving the main thread.
  auto copy_done_callback = [](PJRT_Error *error, void *user_data) {
    EventInstance *dst_buffer_ready_event =
        static_cast<EventInstance *>(user_data);
    delete dst_buffer_ready_event;
    delete ErrorInstance::FromError(error);
  };

  DLOG_F(INFO, "Copy to host id: %d", unique_id());
  tt::runtime::memcpy(dst, tensor());

  auto copy_done_event = new EventInstance();
  copy_done_event->OnReady(copy_done_callback, nullptr);

  *out_done_event = copy_done_event;
  return tt_pjrt_status::kSuccess;
}

PJRT_Buffer_Type BufferInstance::getRuntimeType() {
  DLOG_F(LOG_DEBUG, "BufferInstance::element_type");
  tt::target::DataType Type = tt::runtime::getTensorDataType(tensor());
  return convertElementTypeToBufferType(Type);
}

//===----------------------------------------------------------------------===//
// DeviceDescription
//===----------------------------------------------------------------------===//

DeviceDescription::~DeviceDescription() = default;

void DeviceDescription::BindApi(PJRT_Api *api) {
  DLOG_F(LOG_DEBUG, "DeviceDescription::BindApi");
  api->PJRT_DeviceDescription_Id =
      +[](PJRT_DeviceDescription_Id_Args *args) -> PJRT_Error * {
    args->id = DeviceDescription::Unwrap(args->device_description)->client_id();
    return nullptr;
  };
  api->PJRT_DeviceDescription_ProcessIndex =
      +[](PJRT_DeviceDescription_ProcessIndex_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_ProcessIndex");
    args->process_index =
        DeviceDescription::Unwrap(args->device_description)->process_index();
    return nullptr;
  };
  api->PJRT_DeviceDescription_Attributes =
      +[](PJRT_DeviceDescription_Attributes_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_Attributes");
    // TODO: Implement something.
    args->num_attributes = 0;
    args->attributes = nullptr;
    return nullptr;
  };
  api->PJRT_DeviceDescription_Kind =
      +[](PJRT_DeviceDescription_Kind_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_Kind");
    auto sv =
        DeviceDescription::Unwrap(args->device_description)->kind_string();
    args->device_kind = sv.data();
    args->device_kind_size = sv.size();
    return nullptr;
  };
  api->PJRT_DeviceDescription_DebugString =
      +[](PJRT_DeviceDescription_DebugString_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_DebugString");
    auto sv =
        DeviceDescription::Unwrap(args->device_description)->debug_string();
    args->debug_string = sv.data();
    args->debug_string_size = sv.size();
    return nullptr;
  };
  api->PJRT_DeviceDescription_ToString =
      +[](PJRT_DeviceDescription_ToString_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceDescription::PJRT_DeviceDescription_ToString");
    auto sv = DeviceDescription::Unwrap(args->device_description)->to_string();
    args->to_string = sv.data();
    args->to_string_size = sv.size();
    return nullptr;
  };
}

//===----------------------------------------------------------------------===//
// DeviceInstance
//===----------------------------------------------------------------------===//

DeviceInstance::~DeviceInstance() = default;

void DeviceInstance::BindApi(PJRT_Api *api) {
  DLOG_F(LOG_DEBUG, "DeviceInstance::BindApi");
  api->PJRT_Device_IsAddressable =
      +[](PJRT_Device_IsAddressable_Args *args) -> PJRT_Error * {
    args->is_addressable =
        DeviceInstance::Unwrap(args->device)->is_addressable();
    return nullptr;
  };
  api->PJRT_Device_LocalHardwareId =
      +[](PJRT_Device_LocalHardwareId_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_LocalHardwareId_Args");
    args->local_hardware_id =
        DeviceInstance::Unwrap(args->device)->local_hardware_id();
    return nullptr;
  };
  api->PJRT_Device_AddressableMemories =
      +[](PJRT_Device_AddressableMemories_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_AddressableMemories");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Device_DefaultMemory =
      +[](PJRT_Device_DefaultMemory_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_DefaultMemory");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Device_GetDescription =
      +[](PJRT_Device_GetDescription_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "DeviceInstance::PJRT_Device_GetDescription");
    args->device_description = reinterpret_cast<PJRT_DeviceDescription *>(
        DeviceInstance::Unwrap(args->device)->device_description());
    return nullptr;
  };
}

tt_pjrt_status DeviceInstance::OpenDevice() {
  DLOG_F(LOG_DEBUG, "DeviceInstance::OpenDevice");
  return tt_pjrt_status::kSuccess;
}

tt_pjrt_status DeviceInstance::HostBufferToDevice(
    const void *data, PJRT_Buffer_Type type, const int64_t *dims,
    size_t num_dims, const int64_t *byte_strides, size_t num_byte_strides,
    PJRT_HostBufferSemantics host_buffer_semantics,
    EventInstance **out_done_with_host_buffer_event,
    BufferInstance **out_buffer) {
  DLOG_F(LOG_DEBUG, "DeviceInstance::HostBufferToDevice");

  auto tt_buffer_type = MapBufferTypeToElementType(type);
  tt::target::DataType element_type = tt_buffer_type.first;
  size_t element_size = tt_buffer_type.second;
  std::vector<std::uint32_t> shape;
  std::vector<std::uint32_t> strides;
  if (num_dims == 0) {
    shape.push_back(1);
    strides.push_back(1);
  }
  assert(num_byte_strides == num_dims);
  for (size_t i = 0; i < num_dims; ++i) {
    shape.push_back(dims[i]);
    strides.push_back(byte_strides[i] / element_size);
  }
  std::shared_ptr<void> data_ptr(const_cast<void *>(data), [](void *) {});
  tt::runtime::Tensor tensor = tt::runtime::createTensor(
      data_ptr, shape, strides, element_size, element_type);
  auto buffer_instance = new BufferInstance(*this, tensor, shape, strides);
  DLOG_F(INFO, "Buffer created with id: %d", buffer_instance->unique_id());
  buffer_instance->setType(type);
  *out_buffer = buffer_instance;
  auto event_instance = new EventInstance();
  *out_done_with_host_buffer_event = event_instance;
  return tt_pjrt_status::kSuccess;
}

//===----------------------------------------------------------------------===//
// ClientInstance
//===----------------------------------------------------------------------===//

ClientInstance::ClientInstance(std::unique_ptr<Platform> platform)
    : platform_(std::move(platform)) {
  DLOG_F(LOG_DEBUG, "ClientInstance::ClientInstance");
  module_builder_ = std::make_unique<ModuleBuilder>();
}

ClientInstance::~ClientInstance() {
  DLOG_F(LOG_DEBUG, "ClientInstance::~ClientInstance");
}

PJRT_Error *ClientInstance::Initialize() {
  DLOG_F(LOG_DEBUG, "ClientInstance::Initialize");

  auto status = PopulateDevices();
  if (!tt_pjrt_status_is_ok(status)) {
    return MakeError(status);
  }

  return nullptr;
}

void ClientInstance::BindApi(PJRT_Api *api) {
  // PJRT_Client_Create is polymorphic
  api->PJRT_Client_Destroy =
      +[](PJRT_Client_Destroy_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_Destroy");
    delete ClientInstance::Unwrap(args->client);
    return nullptr;
  };
  api->PJRT_Client_PlatformName =
      +[](PJRT_Client_PlatformName_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_PlatformName");
    auto *client = ClientInstance::Unwrap(args->client);
    args->platform_name = client->cached_platform_name().data();
    args->platform_name_size = client->cached_platform_name().size();
    return nullptr;
  };
  api->PJRT_Client_ProcessIndex =
      +[](PJRT_Client_ProcessIndex_Args *args) -> PJRT_Error * {
    args->process_index = 0;
    return nullptr;
  };
  api->PJRT_Client_PlatformVersion =
      +[](PJRT_Client_PlatformVersion_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_PlatformVersion");
    auto *client = ClientInstance::Unwrap(args->client);
    args->platform_version = client->cached_platform_version().data();
    args->platform_version_size = client->cached_platform_version().size();
    return nullptr;
  };
  api->PJRT_Client_Devices =
      +[](PJRT_Client_Devices_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_Devices");
    auto &devices = ClientInstance::Unwrap(args->client)->devices();
    args->devices = const_cast<PJRT_Device **>(
        reinterpret_cast<PJRT_Device *const *>(devices.data()));
    args->num_devices = devices.size();
    return nullptr;
  };
  api->PJRT_Client_AddressableDevices =
      +[](PJRT_Client_AddressableDevices_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_AddressableDevices_Args");
    auto &devices = ClientInstance::Unwrap(args->client)->addressable_devices();
    args->addressable_devices = const_cast<PJRT_Device **>(
        reinterpret_cast<PJRT_Device *const *>(devices.data()));
    args->num_addressable_devices = devices.size();
    return nullptr;
  };
  api->PJRT_Client_LookupDevice =
      +[](PJRT_Client_LookupDevice_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_LookupDevice_Args");
    auto &devices = ClientInstance::Unwrap(args->client)->devices();
    size_t id_as_size = args->id;
    if (id_as_size >= devices.size()) {
      return MakeError(tt_pjrt_status::kOutOfRange);
    }
    args->device = *devices[id_as_size];
    return nullptr;
  };
  api->PJRT_Client_AddressableMemories =
      +[](PJRT_Client_AddressableMemories_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_AddressableMemories");
    // return MakeError(tt_pjrt_status::kUnimplemented);
    args->num_addressable_memories =
        0; // ClientInstance::Unwrap(args->client)->addressable_memories.size();
    args->addressable_memories =
        nullptr; // ClientInstance::Unwrap(args->client)->addressable_memories.data();
    return nullptr;
  };
  api->PJRT_Client_Compile =
      +[](PJRT_Client_Compile_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_Compile");
    // TODO: It is not great that we only get a client here vs a list of
    // devices to consider (or something). The issue is that systems often
    // have unrelated devices that will not actually be scheduled and those
    // will very naturally have different tuning flags. We therefore have to
    // guess... which is an accident waiting to happen.
    // Looks like what I need is buried in the compile options... need to
    // work on that.
    auto *client = ClientInstance::Unwrap(args->client);
    LoadedExecutableInstance *executable;

    auto *error = client->Compile(args->program, &executable);
    if (error)
      return error;
    args->executable = *executable;
    return nullptr;
  };
  api->PJRT_Client_DefaultDeviceAssignment =
      +[](PJRT_Client_DefaultDeviceAssignment_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_DefaultDeviceAssignment");
    // TODO: Something sensible.
    for (size_t i = 0; i < args->default_assignment_size; ++i) {
      args->default_assignment[i] = 0;
    }
    return nullptr;
  };
  api->PJRT_Client_BufferFromHostBuffer =
      +[](PJRT_Client_BufferFromHostBuffer_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_Client_BufferFromHostBuffer");
    auto status = DeviceInstance::Unwrap(args->device)
                      ->HostBufferToDevice(
                          args->data, args->type, args->dims, args->num_dims,
                          args->byte_strides, args->num_byte_strides,
                          args->host_buffer_semantics,
                          reinterpret_cast<EventInstance **>(
                              &args->done_with_host_buffer),
                          reinterpret_cast<BufferInstance **>(&args->buffer));
    return MakeError(status);
  };
  api->PJRT_LoadedExecutable_Fingerprint =
      +[](PJRT_LoadedExecutable_Fingerprint_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ClientInstance::PJRT_LoadedExecutable_Fingerprint");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
}

tt_pjrt_status ClientInstance::PopulateDevices() {
  DLOG_F(LOG_DEBUG, "ClientInstance::PopulateDevices");
  auto [system_desc, chip_ids] = tt::runtime::getCurrentSystemDesc();
  int device_info_count_ =
      1; // TODO: revert to chip_ids.size(); once
         // https://github.com/tenstorrent/tt-xla/issues/9 is fixed

  devices_.resize(device_info_count_);
  for (size_t i = 0; i < device_info_count_; ++i) {
    devices_[i] = new DeviceInstance(i, *this);
  }

  // For now, just make all devices addressable.
  addressable_devices_.reserve(devices_.size());
  for (auto *device : devices_) {
    addressable_devices_.push_back(device);
  }
  return tt_pjrt_status::kSuccess;
}

PJRT_Error *ClientInstance::Compile(const PJRT_Program *program,
                                    LoadedExecutableInstance **out_executable) {
  DLOG_F(LOG_DEBUG, "ClientInstance::Compile");

  std::string_view code(program->code, program->code_size);
  std::string_view format(program->format, program->format_size);

  tt_pjrt_status status = module_builder_->buildModule(code, format);
  if (!tt_pjrt_status_is_ok(status)) {
    return MakeError(status);
  }

  auto executable = std::make_unique<LoadedExecutableInstance>(
      *this,
      new ExecutableImage(module_builder_->getBinary(),
                          std::string(program->code, program->code_size),
                          module_builder_->getNumInputs(),
                          module_builder_->getNumOutputs()),
      addressable_devices_);
  *out_executable = executable.release();

  return nullptr;
}

std::tuple<uint64_t, uint64_t> ClientInstance::AdvanceTimeline() {
  uint64_t current = execution_timeline_;
  uint64_t next = current + 1;
  execution_timeline_ = next;
  return std::make_tuple(current, next);
}

//===----------------------------------------------------------------------===//
// EventInstance
//===----------------------------------------------------------------------===//

EventInstance::EventInstance() {
  bool fence = false;
  // TODO: fence and wait
  if (!fence) {
    is_ready_ = true;
    return;
  }

  // {
  //   std::lock_guard<std::mutex> guard(lock_);
  //   // Create a thread that waits on the fence and executes the callbacks
  //   when
  //   // the fence is ready.
  //   signal_thread_ = std::make_unique<std::thread>(
  //       [](EventInstance* event_instance,
  //          iree::vm::ref<iree_hal_fence_t> fence) {
  //         iree_status_t wait_status =
  //             iree_hal_fence_wait(fence.get(), iree_infinite_timeout());
  //         event_instance->SignalReady(wait_status);
  //       },
  //       this, std::move(fence));
  // }
}

EventInstance::~EventInstance() {
  std::lock_guard<std::mutex> guard(lock_);
  if (signal_thread_) {
    if (std::this_thread::get_id() != signal_thread_->get_id()) {
      signal_thread_->join();
    } else {
      // An `EventInstance` is allowed to delete itself in one of its callbacks,
      // resulting in `signal_thread_` being the thread calling the destructor.
      // In such cases, we must let the thread continue running independent of
      // the destructor to avoid a deadlock.
      signal_thread_->detach();
      signal_thread_.release();
    }
  }
}

void EventInstance::BindApi(PJRT_Api *api) {
  DLOG_F(LOG_DEBUG, "EventInstance::BindApi");
  api->PJRT_Event_Destroy = +[](PJRT_Event_Destroy_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "EventInstance::PJRT_Event_Destroy");
    auto instance = EventInstance::Unwrap(args->event);
    auto delete_event = [](PJRT_Error *error, void *user_data) {
      EventInstance *event = static_cast<EventInstance *>(user_data);
      delete event;
      if (error) {
        delete ErrorInstance::FromError(error);
      }
    };

    instance->OnReady(delete_event, args->event);
    return nullptr;
  };
  api->PJRT_Event_IsReady = +[](PJRT_Event_IsReady_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "EventInstance::PJRT_Event_IsReady");
    args->is_ready = EventInstance::Unwrap(args->event)->is_ready();
    return nullptr;
  };
  api->PJRT_Event_Error = +[](PJRT_Event_Error_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "EventInstance::PJRT_Event_Error");
    return (PJRT_Error *)EventInstance::Unwrap(args->event)->error();
  };
  api->PJRT_Event_Await = +[](PJRT_Event_Await_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "EventInstance::PJRT_Event_Await");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Event_OnReady = +[](PJRT_Event_OnReady_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "EventInstance::PJRT_Event_OnReady");
    return MakeError(EventInstance::Unwrap(args->event)
                         ->OnReady(args->callback, args->user_arg));
  };
}

ErrorInstance *EventInstance::error() {
  std::lock_guard<std::mutex> guard(lock_);
  if (!tt_pjrt_status_is_ok(status_))
    return new ErrorInstance(status_);
  return nullptr;
}
bool EventInstance::is_ready() {
  DLOG_F(LOG_DEBUG, "EventInstance::is_ready");
  std::lock_guard<std::mutex> guard(lock_);
  return is_ready_;
}

tt_pjrt_status EventInstance::OnReady(PJRT_Event_OnReadyCallback callback,
                                      void *user_arg) {
  DLOG_F(LOG_DEBUG, "EventInstance::OnReady");
  tt_pjrt_status local_status;
  {
    std::lock_guard<std::mutex> guard(lock_);
    if (!is_ready_) {
      pending_callbacks_.push_back({callback, user_arg});
      return tt_pjrt_status::kSuccess;
    }
    local_status = status_;
  }

  // Already signalled. Callback out of lock scope.
  // Note that the callback may destroy the event - so must only operate on
  // locals.
  callback(tt_pjrt_status_is_ok(local_status)
               ? nullptr
               : (PJRT_Error *)new ErrorInstance(local_status),
           user_arg);
  return tt_pjrt_status::kSuccess;
}

void EventInstance::SignalReady(tt_pjrt_status status) {
  DLOG_F(LOG_DEBUG, "EventInstance::SignalReady");
  tt_pjrt_status local_status;
  std::vector<std::pair<PJRT_Event_OnReadyCallback, void *>> local_callbacks;
  {
    std::lock_guard<std::mutex> guard(lock_);
    if (is_ready_) {
      return;
    }
    local_callbacks.swap(pending_callbacks_);
    is_ready_ = true;
    status_ = status;
    local_status = status_;
  }

  // Trigger callbacks outside of the lock.
  // Note that the callback may destroy the event - so must only operate on
  // locals.
  for (auto &cb : local_callbacks) {
    cb.first(tt_pjrt_status_is_ok(local_status)
                 ? nullptr
                 : (PJRT_Error *)new ErrorInstance(local_status),
             cb.second);
  }
}

//===----------------------------------------------------------------------===//
// LoadedExecutableInstance
//===----------------------------------------------------------------------===//

void ExecutableImage::BindApi(PJRT_Api *api) {
  api->PJRT_Executable_Destroy =
      +[](PJRT_Executable_Destroy_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_Destroy");
    ExecutableImage::Unwrap(args->executable)->DecRef();
    return nullptr;
  };
  api->PJRT_Executable_Name =
      +[](PJRT_Executable_Name_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_Name");
    const char *dummy_name = "tt_pjrt_exe";
    args->executable_name = dummy_name;
    args->executable_name_size = std::strlen(dummy_name);
    return nullptr;
  };
  api->PJRT_Executable_SizeOfGeneratedCodeInBytes =
      +[](PJRT_Executable_SizeOfGeneratedCodeInBytes_Args *args)
      -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "ExecutableImage::PJRT_Executable_SizeOfGeneratedCodeInBytes");
    args->size_in_bytes =
        0; // TODO:
           // ExecutableImage::Unwrap(args->executable)->binary->GetDataSize();
    return nullptr;
  };
  api->PJRT_Executable_NumOutputs =
      +[](PJRT_Executable_NumOutputs_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_NumOutputs");
    auto *exec = ExecutableImage::Unwrap(args->executable);
    args->num_outputs = exec->result_count;
    return nullptr;
  };
  api->PJRT_Executable_NumPartitions =
      +[](PJRT_Executable_NumPartitions_Args *args) -> PJRT_Error * {
    // This should be updated once iree supports partitioning.
    args->num_partitions = 1;
    return nullptr;
  };
  api->PJRT_Executable_NumReplicas =
      +[](PJRT_Executable_NumReplicas_Args *args) -> PJRT_Error * {
    // This should be updated once iree supports replicas.
    args->num_replicas = 1;
    return nullptr;
  };
  api->PJRT_Executable_Serialize =
      +[](PJRT_Executable_Serialize_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_Serialize");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_DeserializeAndLoad =
      +[](PJRT_Executable_DeserializeAndLoad_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "ExecutableImage::PJRT_Executable_DeserializeAndLoad_Args");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_Serialize =
      +[](PJRT_Executable_Serialize_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_Serialize_Args");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_OptimizedProgram =
      +[](PJRT_Executable_OptimizedProgram_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_OptimizedProgram");
    ExecutableImage *executable = ExecutableImage::Unwrap(args->executable);
    PJRT_Program *program = args->program;
    program->format = kMlirFormat.data();
    program->format_size = kMlirFormat.size();
    size_t code_size = executable->code.size();
    if (program->code == nullptr) {
      program->code_size = code_size;
    } else {
      if (program->code_size < code_size) {
        return MakeError(tt_pjrt_status::kInvalidArgument);
      }
      std::memcpy(program->code, executable->code.c_str(),
                  executable->code.size());
    }
    return nullptr;
  };
  api->PJRT_Executable_GetCostAnalysis =
      +[](PJRT_Executable_GetCostAnalysis_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_GetCostAnalysis_Args");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_OutputElementTypes =
      +[](PJRT_Executable_OutputElementTypes_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "ExecutableImage::PJRT_Executable_OutputElementTypes_Args");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_OutputDimensions =
      +[](PJRT_Executable_OutputDimensions_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_OutputDimensions_Args");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_Executable_OutputMemoryKinds =
      +[](PJRT_Executable_OutputMemoryKinds_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "ExecutableImage::PJRT_Executable_OutputMemoryKinds");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
}

void LoadedExecutableInstance::BindApi(PJRT_Api *api) {
  api->PJRT_LoadedExecutable_Destroy =
      +[](PJRT_LoadedExecutable_Destroy_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "LoadedExecutableInstance::PJRT_LoadedExecutable_Destroy");
    delete LoadedExecutableInstance::Unwrap(args->executable);
    return nullptr;
  };
  api->PJRT_LoadedExecutable_AddressableDevices =
      +[](PJRT_LoadedExecutable_AddressableDevices_Args *args) -> PJRT_Error * {
    DLOG_F(
        LOG_DEBUG,
        "LoadedExecutableInstance::PJRT_LoadedExecutable_AddressableDevices");
    auto &devices = LoadedExecutableInstance::Unwrap(args->executable)
                        ->addressable_devices();
    args->addressable_devices = const_cast<PJRT_Device **>(
        reinterpret_cast<PJRT_Device *const *>(devices.data()));
    args->num_addressable_devices = devices.size();
    return nullptr;
  };
  api->PJRT_LoadedExecutable_Delete =
      +[](PJRT_LoadedExecutable_Delete_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "LoadedExecutableInstance::PJRT_LoadedExecutable_Delete");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_LoadedExecutable_IsDeleted =
      +[](PJRT_LoadedExecutable_IsDeleted_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "LoadedExecutableInstance::PJRT_LoadedExecutable_IsDeleted_Args");
    return MakeError(tt_pjrt_status::kUnimplemented);
  };
  api->PJRT_LoadedExecutable_Execute =
      +[](PJRT_LoadedExecutable_Execute_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "LoadedExecutableInstance::PJRT_LoadedExecutable_Execute");
    return MakeError(
        LoadedExecutableInstance::Unwrap(args->executable)->Execute(args));
  };
  api->PJRT_LoadedExecutable_GetExecutable =
      +[](PJRT_LoadedExecutable_GetExecutable_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG,
           "LoadedExecutableInstance::PJRT_LoadedExecutable_GetExecutable");
    auto *loaded_exe =
        LoadedExecutableInstance::Unwrap(args->loaded_executable);
    ExecutableImage *image = loaded_exe->image_;

    image->AddRef();
    args->executable = *image;
    return nullptr;
  };
}

tt_pjrt_status
LoadedExecutableInstance::Execute(PJRT_LoadedExecutable_Execute_Args *args) {
  DLOG_F(LOG_DEBUG, "LoadedExecutableInstance::Execute");

  auto [system_desc, chip_ids] = tt::runtime::getCurrentSystemDesc();
  int dev_0 = chip_ids[0];
  auto device = tt::runtime::openDevice({dev_0});

  assert(args->num_devices == 1);
  int dev_index = 0;
  tt::runtime::Binary binary(image_->binary);

  std::vector<tt::runtime::Tensor> rt_inputs;
  rt_inputs.reserve(args->num_args);

  for (size_t i = 0; i < args->num_args; ++i) {
    auto *buffer = BufferInstance::Unwrap(args->argument_lists[dev_index][i]);
    rt_inputs.emplace_back(buffer->tensor());
    DLOG_F(INFO, "Runtime input id: %d", buffer->unique_id());
  }

  std::vector<tt::runtime::Tensor> rt_outputs =
      tt::runtime::submit(device, binary, 0, rt_inputs);
  std::vector<tt::runtime::TensorDesc> output_specs =
      binary.getProgramOutputs(0);

  assert(rt_outputs.size() == output_specs.size());

  for (size_t i = 0; i < output_specs.size(); ++i) {
    bool is_scalar = client_.isOutputScalar(i);
    // PJRT expects an empty shape for scalars.
    std::vector<std::uint32_t> output_shape =
        is_scalar ? std::vector<std::uint32_t>() : output_specs[i].shape;
    auto result_buffer = std::make_unique<BufferInstance>(
        *this->addressable_devices_[dev_index], rt_outputs[i], output_shape,
        output_specs[i].stride);
    result_buffer->setType(
        convertElementTypeToBufferType(output_specs[i].dataType));
    DLOG_F(INFO, "Runtime output id: %d", result_buffer->unique_id());
    args->output_lists[dev_index][i] = *(result_buffer.release());
  }

  if (args->device_complete_events) {
    args->device_complete_events[dev_index] = *(new EventInstance());
  }

  tt::runtime::closeDevice(device);

  return tt_pjrt_status::kSuccess;
}

static void BindUndefineds(PJRT_Api *api) {
#define _STUB(API)                                                             \
  api->API = +[](API##_Args *args) -> decltype(api->API(args)) {               \
    DLOG_F(LOG_DEBUG, "STUB: " #API);                                          \
    return (decltype(api->API(args)))MakeError(                                \
        tt_pjrt_status::kUnimplemented);                                       \
  }

#include "stubs.inc"
}

//===----------------------------------------------------------------------===//
// Top-level API binding.
//===----------------------------------------------------------------------===//

void BindMonomorphicApi(PJRT_Api *api) {
  api->struct_size = PJRT_Api_STRUCT_SIZE;
  api->extension_start = nullptr;
  api->pjrt_api_version.major_version = PJRT_API_MAJOR;
  api->pjrt_api_version.minor_version = PJRT_API_MINOR;

  // This is a bare implementation throwing UNDEFINED errors. This way new
  // functions will not segmentation fault on invocation.
  BindUndefineds(api);
  ErrorInstance::BindApi(api);

  api->PJRT_Plugin_Initialize =
      +[](PJRT_Plugin_Initialize_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "PJRT_Plugin_Initialize");
    return nullptr;
  };

  api->PJRT_Plugin_Attributes = InitializePluginAttributes;

  // Bind by object types.
  BufferInstance::BindApi(api);
  ClientInstance::BindApi(api);
  DeviceDescription::BindApi(api);
  DeviceInstance::BindApi(api);
  EventInstance::BindApi(api);
  ExecutableImage::BindApi(api);
  LoadedExecutableInstance::BindApi(api);
}

PJRT_Error *InitializePluginAttributes(PJRT_Plugin_Attributes_Args *args) {
  DLOG_F(LOG_DEBUG, "PJRT_Plugin_Attributes");

  static std::unique_ptr<PJRTPluginAttributes> s_plugin_attributes =
      std::make_unique<PJRTPluginAttributes>();
  args->attributes = s_plugin_attributes->getAttributes();
  args->num_attributes = s_plugin_attributes->getNumAttributes();

  return nullptr;
}

} // namespace tt::pjrt
