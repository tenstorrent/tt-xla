#include "common/pjrt_implementation/bindings.h"
#include "common/pjrt_implementation/buffer_instance.h"
#include "common/pjrt_implementation/client_instance.h"
#include "common/pjrt_implementation/device_description.h"
#include "common/pjrt_implementation/device_instance.h"
#include "common/pjrt_implementation/error_instance.h"
#include "common/pjrt_implementation/event_instance.h"

namespace tt::pjrt {

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
