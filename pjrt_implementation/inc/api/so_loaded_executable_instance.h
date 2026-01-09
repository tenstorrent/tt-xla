// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// https://llvm.org/LICENSE.txt

#ifndef TT_XLA_PJRT_IMPLEMENTATION_INC_API_SO_LOADED_EXECUTABLE_INSTANCE_H_
#define TT_XLA_PJRT_IMPLEMENTATION_INC_API_SO_LOADED_EXECUTABLE_INSTANCE_H_

// c++ standard library includes
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

// PJRT C API includes
#include "xla/pjrt/c/pjrt_c_api.h"

// tt-mlir includes
#include "mlir/Support/LogicalResult.h"
#include "tt/runtime/types.h"

// tt-xla includes
#include "api/buffer_instance.h"
#include "api/executable_image.h"
#include "api/loaded_executable_instance.h"
#include "utils/status.h"

namespace tt::runtime {
class Device;
class Tensor;
} // namespace tt::runtime

namespace tt::pjrt {

// Class to hold tt-alchemist-python-runner library handle and function
// pointers.
class PythonRunnerHandler {
public:
  // Default constructor leaves the library uninitialized.
  PythonRunnerHandler();

  // Destructor closes the handle to .so file.
  ~PythonRunnerHandler();

  // Initializes the python-runner library and function pointers. This function
  // is fallible.
  void initialize();

  // Getter for initialization status.
  bool isInitialized() const { return m_initialized; }

  // Creates a new PythonModelRunner instance.
  void *createRunner();

  // Destroys a PythonModelRunner instance.
  void destroyRunner(void *runner);

  // Adds a path to Python sys.path.
  void addToSysPath(void *runner, const char *path);

  // Loads a Python module and function.
  void loadModule(void *runner, const char *module_name,
                  const char *function_name);

  // Executes the forward function with given inputs.
  std::vector<tt::runtime::Tensor>
  forward(void *runner, const std::vector<tt::runtime::Tensor> &inputs,
          const tt::runtime::Device &device);

private:
  // Finds python-runner library path using environment variables.
  std::optional<std::string> findPythonRunnerLibraryPath();

  // Initialization status.
  bool m_initialized;

  // The handle to the python-runner .so.
  void *m_handle;

  // Function pointers for PythonModelRunner operations.
  void *(*m_create_runner)();
  void (*m_destroy_runner)(void *);
  void (*m_add_to_sys_path)(void *, const char *);
  void (*m_load_module)(void *, const char *, const char *);
  void (*m_forward)(void *, const tt::runtime::Tensor *, size_t,
                    const tt::runtime::Device *, tt::runtime::Tensor **,
                    size_t *);
};

// Derived class for SO-based loaded executables
class SOLoadedExecutableInstance : public LoadedExecutableInstance {
public:
  // Creates new SO loaded executable instance.
  static std::unique_ptr<SOLoadedExecutableInstance>
  createInstance(std::shared_ptr<SOExecutableImage> executable_image,
                 std::vector<DeviceInstance *> &&addressable_devices,
                 ClientInstance *client_instance);

  // Shares the underlying executable image.
  std::shared_ptr<SOExecutableImage> getSharedExecutableImage() const;

  // Releases the resources this loaded executable uses.
  void releaseResources() override;

  // Runs execution of this loaded executable.
  tt_pjrt_status execute(PJRT_LoadedExecutable_Execute_Args *args) override;

private:
  // Returns an input tensor constructed from the provided buffer instances,
  // prepared for execution. If we cannot reuse the already prepared tensor
  // contained within the buffer instances, this will involve calling
  // `toLayout()` which in most cases involves moving the data to the device.
  std::optional<tt::runtime::Tensor>
  prepareInputTensor(const std::vector<BufferInstance *> &arg_buffers,
                     tt::runtime::Device device, size_t num_devices,
                     std::uint32_t program_index, size_t arg_index) override;

  // Converts input tensor to desired layout. This might move it on device.
  // For SO path, we don't have layout information as we don't have a
  // flatbuffer. HACK: we just return the input tensor (host tensor) without any
  // layout conversion in cases we can't reuse the prepared tensor. This works
  // because codegen code forces layouts (and therefore can basically accept
  // anything).
  tt::runtime::Tensor
  convertTensorLayout(tt::runtime::Tensor input_tensor,
                      std::uint32_t program_index, size_t arg_index,
                      const tt::runtime::Device &runtime_device);

  // Create default-initialized output buffers for SO execution
  void createDefaultOutputBuffers(PJRT_Buffer **const *output_lists,
                                  size_t num_devices);

private:
  // Creates SO loaded executable instance from the executable image.
  SOLoadedExecutableInstance(
      std::shared_ptr<SOExecutableImage> executable_image,
      const std::vector<DeviceInstance *> &addressable_devices,
      ClientInstance *client_instance);
};

} // namespace tt::pjrt

#endif // TT_XLA_PJRT_IMPLEMENTATION_INC_API_SO_LOADED_EXECUTABLE_INSTANCE_H_
