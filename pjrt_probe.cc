#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "xla/pjrt/c/pjrt_c_api.h"

static void print_error(PJRT_Api *api, PJRT_Error *err) {
  if (!err)
    return;
  PJRT_Error_Message_Args msg_args{};
  msg_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  msg_args.error = err;
  api->PJRT_Error_Message(&msg_args);
  std::cerr << "PJRT_Error: "
            << std::string(msg_args.message, msg_args.message_size) << "\n";
  PJRT_Error_Destroy_Args destroy_args{};
  destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  destroy_args.error = err;
  api->PJRT_Error_Destroy(&destroy_args);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " /path/to/libpjrt_plugin_tt.so\n";
    return 1;
  }

  void *handle = dlopen(argv[1], RTLD_NOW);
  if (!handle) {
    std::cerr << "dlopen failed: " << dlerror() << "\n";
    return 1;
  }

  using GetPjrtApiVersionFn = unsigned (*)();
  using GetPjrtApiFn = PJRT_Api *(*)();

  auto get_version = (GetPjrtApiVersionFn)dlsym(handle, "GetPjrtApiVersion");
  auto get_api = (GetPjrtApiFn)dlsym(handle, "GetPjrtApi");
  if (!get_version || !get_api) {
    std::cerr << "dlsym failed for GetPjrtApiVersion/GetPjrtApi\n";
    return 1;
  }

  std::cout << "PJRT plugin API version: " << get_version() << "\n";
  PJRT_Api *api = get_api();

  // Plugin init
  PJRT_Plugin_Initialize_Args init_args{};
  init_args.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
  print_error(api, api->PJRT_Plugin_Initialize(&init_args));

  // Create client
  PJRT_Client_Create_Args create_args{};
  create_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  print_error(api, api->PJRT_Client_Create(&create_args));
  PJRT_Client *client = create_args.client;

  // Platform info
  PJRT_Client_PlatformName_Args name_args{};
  name_args.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  name_args.client = client;
  print_error(api, api->PJRT_Client_PlatformName(&name_args));
  std::cout << "platform_name: "
            << std::string(name_args.platform_name,
                           name_args.platform_name_size)
            << "\n";

  PJRT_Client_PlatformVersion_Args ver_args{};
  ver_args.struct_size = PJRT_Client_PlatformVersion_Args_STRUCT_SIZE;
  ver_args.client = client;
  print_error(api, api->PJRT_Client_PlatformVersion(&ver_args));
  std::cout << "platform_version: "
            << std::string(ver_args.platform_version,
                           ver_args.platform_version_size)
            << "\n";

  // Devices
  PJRT_Client_Devices_Args devices_args{};
  devices_args.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
  devices_args.client = client;
  print_error(api, api->PJRT_Client_Devices(&devices_args));
  std::cout << "num_devices: " << devices_args.num_devices << "\n";

  if (devices_args.num_devices > 0) {
    // device_description
    std::cout << "---- device_description ----\n";

    PJRT_Device *device = devices_args.devices[0];
    PJRT_Device_GetDescription_Args desc_args{};
    desc_args.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
    desc_args.device = device;
    print_error(api, api->PJRT_Device_GetDescription(&desc_args));

    PJRT_DeviceDescription_Id_Args id_args{};
    id_args.struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE;
    id_args.device_description = desc_args.device_description;
    print_error(api, api->PJRT_DeviceDescription_Id(&id_args));
    std::cout << "device_id: " << id_args.id << "\n";

    PJRT_DeviceDescription_ProcessIndex_Args proc_args{};
    proc_args.struct_size =
        PJRT_DeviceDescription_ProcessIndex_Args_STRUCT_SIZE;
    proc_args.device_description = desc_args.device_description;
    print_error(api, api->PJRT_DeviceDescription_ProcessIndex(&proc_args));
    std::cout << "device_process_index: " << proc_args.process_index << "\n";

    PJRT_DeviceDescription_Attributes_Args attr_args{};
    attr_args.struct_size = PJRT_DeviceDescription_Attributes_Args_STRUCT_SIZE;
    attr_args.device_description = desc_args.device_description;
    print_error(api, api->PJRT_DeviceDescription_Attributes(&attr_args));
    std::cout << "device_attributes: " << attr_args.num_attributes << "\n";
    for (size_t i = 0; i < attr_args.num_attributes; ++i) {
      const PJRT_NamedValue &nv = attr_args.attributes[i];
      std::string name(nv.name, nv.name_size);
      std::cout << "  " << name << " = ";
      switch (nv.type) {
      case PJRT_NamedValue_kString:
        std::cout << std::string(nv.string_value, nv.value_size);
        break;
      case PJRT_NamedValue_kInt64:
        std::cout << nv.int64_value;
        break;
      case PJRT_NamedValue_kInt64List: {
        std::cout << "[";
        for (size_t j = 0; j < nv.value_size; ++j) {
          if (j)
            std::cout << ", ";
          std::cout << nv.int64_array_value[j];
        }
        std::cout << "]";
        break;
      }
      case PJRT_NamedValue_kFloat:
        std::cout << nv.float_value;
        break;
      case PJRT_NamedValue_kBool:
        std::cout << (nv.bool_value ? "true" : "false");
        break;
      default:
        std::cout << "<unknown>";
        break;
      }
      std::cout << "\n";
    }

    PJRT_DeviceDescription_Kind_Args kind_args{};
    kind_args.struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE;
    kind_args.device_description = desc_args.device_description;
    print_error(api, api->PJRT_DeviceDescription_Kind(&kind_args));
    std::cout << "device_kind: "
              << std::string(kind_args.device_kind, kind_args.device_kind_size)
              << "\n";

    PJRT_DeviceDescription_DebugString_Args dbg_args{};
    dbg_args.struct_size = PJRT_DeviceDescription_DebugString_Args_STRUCT_SIZE;
    dbg_args.device_description = desc_args.device_description;
    print_error(api, api->PJRT_DeviceDescription_DebugString(&dbg_args));
    std::cout << "device_debug_string: "
              << std::string(dbg_args.debug_string, dbg_args.debug_string_size)
              << "\n";

    PJRT_DeviceDescription_ToString_Args to_args{};
    to_args.struct_size = PJRT_DeviceDescription_ToString_Args_STRUCT_SIZE;
    to_args.device_description = desc_args.device_description;
    print_error(api, api->PJRT_DeviceDescription_ToString(&to_args));
    std::cout << "device_to_string: "
              << std::string(to_args.to_string, to_args.to_string_size) << "\n";

    // device_instance
    std::cout << "---- device_instance ----\n";

    PJRT_Device_GetDescription_Args dev_desc_args{};
    dev_desc_args.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
    dev_desc_args.device = device;
    print_error(api, api->PJRT_Device_GetDescription(&dev_desc_args));

    PJRT_DeviceDescription_ToString_Args dev_desc_str_args{};
    dev_desc_str_args.struct_size =
        PJRT_DeviceDescription_ToString_Args_STRUCT_SIZE;
    dev_desc_str_args.device_description = dev_desc_args.device_description;
    print_error(api, api->PJRT_DeviceDescription_ToString(&dev_desc_str_args));
    std::cout << "device_description: "
              << std::string(dev_desc_str_args.to_string,
                             dev_desc_str_args.to_string_size)
              << "\n";

    PJRT_Device_IsAddressable_Args addr_args{};
    addr_args.struct_size = PJRT_Device_IsAddressable_Args_STRUCT_SIZE;
    addr_args.device = device;
    print_error(api, api->PJRT_Device_IsAddressable(&addr_args));
    std::cout << "device_is_addressable: "
              << (addr_args.is_addressable ? "true" : "false") << "\n";

    PJRT_Device_LocalHardwareId_Args hw_args{};
    hw_args.struct_size = PJRT_Device_LocalHardwareId_Args_STRUCT_SIZE;
    hw_args.device = device;
    print_error(api, api->PJRT_Device_LocalHardwareId(&hw_args));
    std::cout << "device_local_hardware_id: " << hw_args.local_hardware_id
              << "\n";

    PJRT_Device_AddressableMemories_Args mems_args{};
    mems_args.struct_size = PJRT_Device_AddressableMemories_Args_STRUCT_SIZE;
    mems_args.device = device;
    print_error(api, api->PJRT_Device_AddressableMemories(&mems_args));
    std::cout << "device_addressable_memories: " << mems_args.num_memories
              << "\n";

    PJRT_Device_DefaultMemory_Args defmem_args{};
    defmem_args.struct_size = PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
    defmem_args.device = device;
    print_error(api, api->PJRT_Device_DefaultMemory(&defmem_args));
    std::cout << "device_default_memory: " << defmem_args.memory << "\n";

    // client_instance
    std::cout << "---- client_instance ----\n";

    PJRT_Client_ProcessIndex_Args proc_idx_args{};
    proc_idx_args.struct_size = PJRT_Client_ProcessIndex_Args_STRUCT_SIZE;
    proc_idx_args.client = client;
    print_error(api, api->PJRT_Client_ProcessIndex(&proc_idx_args));
    std::cout << "client_process_index: " << proc_idx_args.process_index
              << "\n";

    PJRT_Client_AddressableDevices_Args addr_devs_args{};
    addr_devs_args.struct_size =
        PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;
    addr_devs_args.client = client;
    print_error(api, api->PJRT_Client_AddressableDevices(&addr_devs_args));
    std::cout << "client_addressable_devices: "
              << addr_devs_args.num_addressable_devices << "\n";

    PJRT_Client_LookupDevice_Args lookup_dev_args{};
    lookup_dev_args.struct_size = PJRT_Client_LookupDevice_Args_STRUCT_SIZE;
    lookup_dev_args.client = client;
    lookup_dev_args.id = id_args.id;
    print_error(api, api->PJRT_Client_LookupDevice(&lookup_dev_args));
    std::cout << "client_lookup_device ptr: " << lookup_dev_args.device << "\n";

    PJRT_Client_LookupAddressableDevice_Args lookup_addr_dev_args{};
    lookup_addr_dev_args.struct_size =
        PJRT_Client_LookupAddressableDevice_Args_STRUCT_SIZE;
    lookup_addr_dev_args.client = client;
    lookup_addr_dev_args.local_hardware_id = hw_args.local_hardware_id;
    print_error(
        api, api->PJRT_Client_LookupAddressableDevice(&lookup_addr_dev_args));
    std::cout << "client_lookup_addressable_device ptr: "
              << lookup_addr_dev_args.addressable_device << "\n";

    PJRT_Client_AddressableMemories_Args addr_mems_args{};
    addr_mems_args.struct_size =
        PJRT_Client_AddressableMemories_Args_STRUCT_SIZE;
    addr_mems_args.client = client;
    print_error(api, api->PJRT_Client_AddressableMemories(&addr_mems_args));
    std::cout << "client_addressable_memories: "
              << addr_mems_args.num_addressable_memories << "\n";

    // Default device assignment (replicas=1, partitions=1)
    int default_assignment[1] = {-1};
    PJRT_Client_DefaultDeviceAssignment_Args assign_args{};
    assign_args.struct_size =
        PJRT_Client_DefaultDeviceAssignment_Args_STRUCT_SIZE;
    assign_args.client = client;
    assign_args.num_replicas = 1;
    assign_args.num_partitions = 1;
    assign_args.default_assignment_size = 1;
    assign_args.default_assignment = default_assignment;
    print_error(api, api->PJRT_Client_DefaultDeviceAssignment(&assign_args));
    std::cout << "client_default_device_assignment[0]: "
              << default_assignment[0] << "\n";

    // BufferFromHostBuffer (tiny 2x2 f32) + cleanup
    float host_data[4] = {1.f, 2.f, 3.f, 4.f};
    int64_t dims[2] = {2, 2};
    int64_t byte_strides[2] = {static_cast<int64_t>(sizeof(float) * 2),
                               static_cast<int64_t>(sizeof(float))};

    PJRT_Client_BufferFromHostBuffer_Args buf_args{};
    buf_args.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    buf_args.client = client;
    buf_args.data = host_data;
    buf_args.type = PJRT_Buffer_Type_F32;
    buf_args.dims = dims;
    buf_args.num_dims = 2;
    buf_args.byte_strides = byte_strides;
    buf_args.num_byte_strides = 2;
    buf_args.host_buffer_semantics =
        PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    buf_args.device = device;
    buf_args.memory = defmem_args.memory; // or nullptr to use device default
    buf_args.device_layout = nullptr;

    print_error(api, api->PJRT_Client_BufferFromHostBuffer(&buf_args));
    std::cout << "client_buffer_from_host_buffer: buffer=" << buf_args.buffer
              << " event=" << buf_args.done_with_host_buffer << "\n";

    if (buf_args.buffer) {
      PJRT_Buffer_Destroy_Args buf_destroy{};
      buf_destroy.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
      buf_destroy.buffer = buf_args.buffer;
      print_error(api, api->PJRT_Buffer_Destroy(&buf_destroy));
    }
    if (buf_args.done_with_host_buffer) {
      PJRT_Event_Destroy_Args evt_destroy{};
      evt_destroy.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
      evt_destroy.event = buf_args.done_with_host_buffer;
      print_error(api, api->PJRT_Event_Destroy(&evt_destroy));
    }

    const char *mlir_path = std::getenv("PJRT_PROBE_MLIR_PATH");
    if (mlir_path) {
      std::ifstream file(mlir_path, std::ios::in | std::ios::binary);
      std::string mlir((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());

      PJRT_Program program{};
      program.struct_size = PJRT_Program_STRUCT_SIZE;
      program.code = mlir.data();
      program.code_size = mlir.size();
      program.format = "mlir";
      program.format_size = 4;

      PJRT_Client_Compile_Args compile_args{};
      compile_args.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
      compile_args.client = client;
      compile_args.program = &program;
      compile_args.compile_options = nullptr;
      compile_args.compile_options_size = 0;

      print_error(api, api->PJRT_Client_Compile(&compile_args));

      if (compile_args.executable) {
        PJRT_LoadedExecutable_GetExecutable_Args get_exec_args{};
        get_exec_args.struct_size =
            PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
        get_exec_args.loaded_executable = compile_args.executable;
        print_error(api,
                    api->PJRT_LoadedExecutable_GetExecutable(&get_exec_args));

        PJRT_Executable_Name_Args name_args{};
        name_args.struct_size = PJRT_Executable_Name_Args_STRUCT_SIZE;
        name_args.executable = get_exec_args.executable;
        print_error(api, api->PJRT_Executable_Name(&name_args));
        std::cout << "executable_name: "
                  << std::string(name_args.executable_name,
                                 name_args.executable_name_size)
                  << "\n";

        PJRT_Executable_Destroy_Args exec_destroy{};
        exec_destroy.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
        exec_destroy.executable = get_exec_args.executable;
        print_error(api, api->PJRT_Executable_Destroy(&exec_destroy));

        PJRT_LoadedExecutable_Destroy_Args loaded_destroy{};
        loaded_destroy.struct_size =
            PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        loaded_destroy.executable = compile_args.executable;
        print_error(api, api->PJRT_LoadedExecutable_Destroy(&loaded_destroy));
      }
    } else {
      std::cout << "executable_name: skipped (set PJRT_PROBE_MLIR_PATH)\n";
    }

  } else {
    std::cout << "no devices found\n";
  }

  // Cleanup
  PJRT_Client_Destroy_Args destroy_args{};
  destroy_args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  destroy_args.client = client;
  print_error(api, api->PJRT_Client_Destroy(&destroy_args));

  dlclose(handle);
  return 0;
}
