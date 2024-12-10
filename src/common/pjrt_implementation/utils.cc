#include "common/pjrt_implementation/utils.h"

#include "common/pjrt_implementation/buffer_instance.h"
#include "common/pjrt_implementation/client_instance.h"
#include "common/pjrt_implementation/device_description.h"
#include "common/pjrt_implementation/device_instance.h"
#include "common/pjrt_implementation/error_instance.h"
#include "common/pjrt_implementation/event_instance.h"

namespace tt::pjrt {

PJRT_Buffer_Type
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

static void BindUndefineds(PJRT_Api *api) {
#define _STUB(API)                                                             \
  api->API = +[](API##_Args *args) -> decltype(api->API(args)) {               \
    DLOG_F(LOG_DEBUG, "STUB: " #API);                                          \
    return (decltype(api->API(args)))MakeError(                                \
        tt_pjrt_status::kUnimplemented);                                       \
  }

#include "stubs.inc"
}

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

  api->PJRT_Plugin_Attributes =
      +[](PJRT_Plugin_Attributes_Args *args) -> PJRT_Error * {
    DLOG_F(LOG_DEBUG, "PJRT_Plugin_Attributes");
    args->num_attributes = 0;
    return nullptr;
  };

  // Bind by object types.
  BufferInstance::BindApi(api);
  ClientInstance::BindApi(api);
  DeviceDescription::BindApi(api);
  DeviceInstance::BindApi(api);
  EventInstance::BindApi(api);
  ExecutableImage::BindApi(api);
  LoadedExecutableInstance::BindApi(api);
}

} // namespace tt::pjrt
