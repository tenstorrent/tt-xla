

# ----- TTXLADebug -----
# Static lib that implements some bindings to TTMLIRRuntime to enable easier debug
# Depends on: TTMLIRRuntime

if (TT_RUNTIME_DEBUG)
    add_compile_definitions(TT_RUNTIME_DEBUG=1)
endif()

# Ensure RTTI is enabled for the TTXLADebug pybind11 module
set_source_files_properties(bindings.cc PROPERTIES COMPILE_FLAGS "-frtti")

list(APPEND CMAKE_PREFIX_PATH "${TTPJRT_SOURCE_DIR}/venv/lib/python3.11/site-packages/pybind11" "/usr/local/lib/python3.11/dist-packages/pybind11")
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)


if(EXISTS "${TTPJRT_SOURCE_DIR}/venv/lib/python3.11/site-packages/torch")
    set(TORCH_INSTALL_PREFIX "${TTPJRT_SOURCE_DIR}/venv/lib/python3.11/site-packages/torch")
elseif(EXISTS "/usr/local/lib/python3.11/dist-packages/torch")
    set(TORCH_INSTALL_PREFIX "/usr/local/lib/python3.11/dist-packages/torch")
else()
    message(FATAL_ERROR "Could not find torch installation in either '${TTPJRT_SOURCE_DIR}/venv/lib/python3.11/site-packages/torch' or '/usr/local/lib/python3.11/dist-packages/torch'")
endif()

set(CMAKE_PREFIX_PATH ${TORCH_INSTALL_PREFIX})

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")

pybind11_add_module(TTXLADebug bindings.cc)

target_include_directories(TTXLADebug PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/install/include
    ${TTMLIR_TOOLCHAIN_DIR}/include
    ${TORCH_INCLUDE_DIRS}
)

add_dependencies(TTXLADebug
    tt-mlir
)

target_link_libraries(TTXLADebug
PRIVATE
    TTMLIRRuntime
PUBLIC
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    coverage_config
)

target_link_directories(TTXLADebug PUBLIC
    ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/install/lib
)

set_target_properties(TTXLADebug
    PROPERTIES
    PREFIX ""  # Disable "lib" prefix.
    LIBRARY_OUTPUT_NAME tt_xla_debug
    BUILD_RPATH "$ORIGIN:$ORIGIN/lib"
    INSTALL_RPATH "$ORIGIN:$ORIGIN/lib"
)

install(TARGETS TTXLADebug DESTINATION ${CMAKE_INSTALL_PREFIX})

add_custom_target(install-tt_xla_debug ALL
    COMMAND ln -sf $<TARGET_FILE:TTXLADebug> ${CMAKE_SOURCE_DIR}/python_package/tt_xla_debug/tt_xla_debug.so
    COMMENT "Installing tt_xla_debug.so to python_package/tt_xla_debug"
    COMMAND pip install -e ${CMAKE_SOURCE_DIR}/python_package --no-deps --no-build-isolation
    COMMENT "Running editable installation..."
    USES_TERMINAL
)

add_dependencies(install-tt_xla_debug TTXLADebug)
