// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <torch/extension.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
#include "tt/runtime/debug.h"
#endif

#include "tt/runtime/runtime.h"
#include "tt/runtime/types.h"
#include "tt/runtime/utils.h"
namespace py = pybind11;

static at::ScalarType dt_to_torch_scalar_type(tt::target::DataType df) {
  switch (df) {
  case tt::target::DataType::UInt8:
    return at::ScalarType::Byte;
  case tt::target::DataType::UInt16:
    return at::ScalarType::Short;
  case tt::target::DataType::UInt32:
    return at::ScalarType::UInt32;
  case tt::target::DataType::Int32:
    return at::ScalarType::Int;
  case tt::target::DataType::Float16:
    return at::ScalarType::Half;
  case tt::target::DataType::Float32:
    return at::ScalarType::Float;
  case tt::target::DataType::BFloat16:
    return at::ScalarType::BFloat16;
  default:
    break;
  }
  assert(false && "Unsupported scalar type");
  return at::ScalarType::Undefined;
}

static tt::target::DataType torch_scalar_type_to_dt(at::ScalarType st) {
  switch (st) {
  case at::ScalarType::Byte:
    return tt::target::DataType::UInt8;
  case at::ScalarType::Char:
    return tt::target::DataType::UInt8;
  case at::ScalarType::Short:
    return tt::target::DataType::UInt16;
  case at::ScalarType::UInt32:
    return tt::target::DataType::UInt32;
  case at::ScalarType::Int:
    return tt::target::DataType::Int32;
  case at::ScalarType::Long:
    return tt::target::DataType::UInt32;
  case at::ScalarType::Half:
    return tt::target::DataType::Float16;
  case at::ScalarType::Float:
    return tt::target::DataType::Float32;
  // case torch::ScalarType::Double:
  // case torch::ScalarType::ComplexHalf:
  // case torch::ScalarType::ComplexFloat:
  // case torch::ScalarType::ComplexDouble:
  // case torch::ScalarType::Bool:
  case at::ScalarType::BFloat16:
    return tt::target::DataType::BFloat16;
  case at::ScalarType::Bool:
    // tt-metal does not support boolean data type; so bfloat16 data type is
    // used instead.
    return tt::target::DataType::BFloat16;
  default:
    break;
  }
  assert(false && "Unsupported scalar type");
  return tt::target::DataType::UInt8;
}

template <typename T>
std::vector<int64_t> as_vec_int64(std::vector<T> const &vec) {
  std::vector<int64_t> result;
  result.reserve(vec.size());
  for (auto const &v : vec) {
    result.push_back(v);
  }
  return result;
}

static tt::runtime::Tensor create_tensor(const at::Tensor &tensor) {
  auto shape =
      std::vector<uint32_t>(tensor.sizes().begin(), tensor.sizes().end());
  if (shape.empty()) {
    shape.push_back(1);
  }

  assert(tensor.is_contiguous() && "Cannot create runtime tensor from "
                                   "non-contiguous torch tensor");

  // Torch tensors which are contiguous may not always have a stride
  // attribute which indicates that the tensor is contiguous. This occurs
  // when the left-most dimension is 1. In such cases, when the left-most
  // dimension is 1, the stride value for that dimension will never be used,
  // and so they do not bother to compute it even when calling .contiguous().
  //
  // Our runtime expects that this stride is accurate. So, we will require
  // that this torch tensor is contiguous and then calculate a fully-accurate
  // stride for it.
  std::vector<uint32_t> stride = tt::runtime::utils::calculateStride(shape);

  // Check if a tensor is empty using its shape.
  bool isEmptyTensor = std::any_of(shape.begin(), shape.end(),
                                   [](uint32_t x) { return x == 0; });

  // Return a host owned tensor if tensor is empty; otherwise return host
  // borrowed tensor.
  return isEmptyTensor
             ? tt::runtime::createOwnedHostTensor(
                   tensor.data_ptr(), shape, stride, tensor.element_size(),
                   torch_scalar_type_to_dt(tensor.scalar_type()))
             : tt::runtime::createBorrowedHostTensor(
                   tensor.data_ptr(), shape, stride, tensor.element_size(),
                   torch_scalar_type_to_dt(tensor.scalar_type()));
}

static at::Tensor create_torch_tensor(const tt::runtime::Tensor &tensor) {
  tt::runtime::Tensor untilized_tensor =
      tt::runtime::toHost(tensor, /*untilize=*/true)[0];

  const std::vector<std::int64_t> shape =
      as_vec_int64(tt::runtime::getTensorShape(untilized_tensor));
  const std::vector<std::int64_t> stride =
      as_vec_int64(tt::runtime::getTensorStride(untilized_tensor));

  const tt::target::DataType rt_datatype =
      tt::runtime::getTensorDataType(untilized_tensor);
  const at::ScalarType dataType = dt_to_torch_scalar_type(rt_datatype);

  at::Tensor torch_tensor =
      at::empty(shape, at::TensorOptions().dtype(dataType))
          .as_strided(shape, stride);
  tt::runtime::Tensor rt_tensor = create_tensor(torch_tensor);
  tt::runtime::memcpy(rt_tensor, untilized_tensor);

  return torch_tensor;
}

at::Tensor
get_op_output_torch_tensor(tt::runtime::OpContext opContextHandle,
                           tt::runtime::CallbackContext programContextHandle) {

  auto tensorMap =
      tt::runtime::getOpOutputTensor(opContextHandle, programContextHandle);

  // Some ops in a decomposed tfx node may not have valid output tensors (eg.
  // deallocate) For these, return an empty tensor

  if (tensorMap.empty()) {
    std::cout << "Warning: getOpOutputTensor does not return any tensor."
              << std::endl;
    return at::Tensor(); // Return an empty PyTorch tensor
  }

  // Return the first tensor in the map. We do not currently support
  // intermediate comparison for ops with multiple outputs
  tt::runtime::Tensor tensor = tensorMap.begin()->second;

  return create_torch_tensor(tensor);
}

PYBIND11_MODULE(tt_xla_debug, m) {
#if defined(TT_RUNTIME_DEBUG) && TT_RUNTIME_DEBUG == 1
  py::class_<tt::runtime::CallbackContext>(m, "CallbackContext");
  py::class_<tt::runtime::OpContext>(m, "OpContext");
  py::class_<tt::runtime::TensorDesc>(m, "TensorDesc")
      .def_readonly("shape", &tt::runtime::TensorDesc::shape)
      .def_readonly("stride", &tt::runtime::TensorDesc::stride)
      .def_readonly("itemsize", &tt::runtime::TensorDesc::itemsize)
      .def_readonly("dataType", &tt::runtime::TensorDesc::dataType);
  py::class_<tt::runtime::Binary>(m, "Binary")
      .def("getProgramInputs", &tt::runtime::Binary::getProgramInputs)
      .def("getProgramOutputs", &tt::runtime::Binary::getProgramOutputs)
      .def("asJson", &tt::runtime::Binary::asJson);
  m.def("get_op_output_tensor", &tt::runtime::getOpOutputTensor);
  m.def("get_op_output_tensor_desc", &tt::runtime::getTensorDesc);
  m.def("get_op_output_torch_tensor", &get_op_output_torch_tensor);
  m.def("get_op_debug_str", &tt::runtime::getOpDebugString,
        "Get the debug string of the op");
  m.def("get_op_loc_info", &tt::runtime::getOpLocInfo,
        "Get the location info of the op");
  py::class_<tt::runtime::debug::Hooks>(m, "DebugHooks")
      .def_static(
          "get_debug_hooks",
          [](py::function func) {
            func.inc_ref();
            auto holder =
                std::shared_ptr<PyObject>(func.ptr(), [](PyObject *obj) {
                  py::gil_scoped_acquire gil;
                  Py_DECREF(obj);
                });

            return tt::runtime::debug::Hooks::get(
                std::nullopt,
                [holder](tt::runtime::Binary binary,
                         tt::runtime::CallbackContext callback_ctx,
                         tt::runtime::OpContext op_ctx) {
                  py::gil_scoped_acquire gil;
                  auto callable =
                      py::reinterpret_borrow<py::function>(holder.get());
                  callable(binary, callback_ctx, op_ctx);
                });
          },
          "Get the debug hooks")
      .def("__str__", [](const tt::runtime::debug::Hooks &hooks) {
        std::stringstream os;
        os << hooks;
        return os.str();
      });

  /**
   * Cleanup code to force a well ordered destruction w.r.t. the GIL
   */
  auto cleanup_callback = []() {
    tt::runtime::debug::Hooks::get().unregisterHooks();
  };
  m.add_object("_cleanup", py::capsule(cleanup_callback));
  m.def("unregister_hooks",
        []() { tt::runtime::debug::Hooks::get().unregisterHooks(); });
  m.def("is_runtime_debug_enabled", []() -> bool { return true; });
#else
  m.def("is_runtime_debug_enabled", []() -> bool { return false; });
#endif
}
