# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# This file incorporates work covered by the following copyright and permission notice:
# SPDX-FileCopyrightText: Copyright 2023 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# https://llvm.org/LICENSE.txt

# ----- TTPJRTApi -----
# Static lib containing core PJRT API implementation: handling device/client/buffer/etc.
# Depends on: TTMLIRCompiler, TTMLIRRuntime, TTPJRTUtils, coverage_config, protobuf::libprotobuf,
#             Python::Module, pybind11::embed (for the embedded tt-lang resolve call).

find_package(Protobuf REQUIRED)

# Python interpreter + headers + libpython, used by
# ModuleBuilder::resolveTtLangKernels to call into
# `tt_torch.tt_lang.resolve_kernel`. We pull in `Development` (a superset of
# Development.Module + Development.Embed) because pybind11::embed requires
# the libpython link to resolve `Py_*` symbols. The plugin is always loaded
# by a running Python process (JAX/PyTorch), so at runtime DT_NEEDED for
# libpython is satisfied by the host's already-loaded copy and we never call
# Py_Initialize() ourselves; we just take the GIL on the existing
# interpreter.
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Locate pybind11 in the active Python environment unless the caller has
# already pointed CMake at one. This makes the configure step Just Work inside
# the tt-xla venv (which has pybind11 pip-installed) without any extra flags.
if(NOT pybind11_DIR)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE _tt_pybind11_cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _tt_pybind11_rc
    )
    if(_tt_pybind11_rc EQUAL 0 AND _tt_pybind11_cmakedir)
        set(pybind11_DIR "${_tt_pybind11_cmakedir}" CACHE PATH "pybind11 CMake config dir")
    endif()
endif()
find_package(pybind11 CONFIG REQUIRED)

add_library(TTPJRTApi
    "buffer_instance.cc"
    "client_instance.cc"
    "device_description.cc"
    "device_instance.cc"
    "error_instance.cc"
    "event_instance.cc"
    "executable_image.cc"
    "executable_instance.cc"
    "loaded_executable_instance.cc"
    "flatbuffer_loaded_executable_instance.cc"
    "so_loaded_executable_instance.cc"
    "memory_instance.cc"
    "compile_options.cc"
    "compile_options_parser.cc"
    "module_builder/frontend_passes/shlo_clean_for_xla_ingestion.cc"
    "module_builder/frontend_passes/shlo_input_role_propagation.cc"
    "module_builder/frontend_passes/shlo_set_proper_sdy_mesh_attribute.cc"
    "module_builder/module_builder.cc"
    "module_builder/tt_lang_bridge.cc"
    "plugin_attributes.cc"
    "serialized_executable_instance.cc"
    "tensor.cc"
    "tensor_pool.cc"
)

target_include_directories(TTPJRTApi
PRIVATE
    # Includes used in .cpp files.

    # Used for tracy zones in PJRT.
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/tt_metal/third_party/tracy/public
PUBLIC
    # Includes used in public headers.
    ${PROJECT_SOURCE_DIR}/pjrt_implementation/inc
    ${TTMLIR_TOOLCHAIN_DIR}/src/shardy
    ${TTMLIR_TOOLCHAIN_DIR}/src/stablehlo
    ${TTMLIR_TOOLCHAIN_DIR}/include
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/install/include
)

target_link_libraries(TTPJRTApi
PRIVATE
    dl
    coverage_config
    protobuf::libprotobuf
    TTMLIRCompiler
    TTMLIRRuntime
PUBLIC
    TTPJRTUtils
    # pybind11::embed pulls in libpython so we can call into the user's Python
    # process from the tt-lang bridge. We do NOT start a new interpreter; we
    # attach to the one that loaded the plugin. PUBLIC (rather than PRIVATE)
    # because TTPJRTApi is a STATIC library -- without propagation, the final
    # plugin .so wouldn't see the libpython link requirement and would fail to
    # link with undefined Py_* symbols.
    pybind11::embed
)

if(TTXLA_ENABLE_EMITPY_EXECUTION)
    target_link_libraries(TTPJRTApi PRIVATE tt-alchemist-python-runner)
endif()

target_compile_definitions(TTPJRTApi PRIVATE TTXLA_ENABLE_EMITPY_EXECUTION=$<BOOL:${TTXLA_ENABLE_EMITPY_EXECUTION}>)

# pybind11 uses typeid() pervasively, so the embedded-Python bridge must be
# built with RTTI. The rest of TTPJRTApi inherits -fno-rtti from LLVM/MLIR
# and we want to keep it that way -- so we apply -frtti only to the bridge
# translation unit.
set_source_files_properties(
    "module_builder/tt_lang_bridge.cc"
    PROPERTIES COMPILE_OPTIONS "-frtti"
)

# Enable Tracy profiling if requested
if(TTXLA_TRACY_ZONES)
    # Define TRACY_ENABLE to enable Tracy zones in the code - this is used in `tracy` headers.
    target_compile_definitions(TTPJRTApi PUBLIC TRACY_ENABLE)
    target_link_libraries(TTPJRTApi PUBLIC tracy)
    message(STATUS "PJRT tracy zones enabled")
endif()

target_link_directories(TTPJRTApi PUBLIC
    ${TTMLIR_LIB_DIR}
)

set_target_properties(TTPJRTApi PROPERTIES
    UNIT_TESTS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/unit_tests"
)
