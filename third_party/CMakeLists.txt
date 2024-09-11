include(ExternalProject)
add_subdirectory(pjrt_c_api)

set(TTMLIR_LIB_DIR ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/lib/SharedLib)
ExternalProject_Add(
    tt-mlir
    PREFIX ${TTPJRT_SOURCE_DIR}/third_party/tt-mlir
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_C_COMPILER=clang-17
      -DCMAKE_CXX_COMPILER=clang++-17
      -DTT_RUNTIME_ENABLE_TTNN=ON
      -DTTMLIR_ENABLE_STABLEHLO=ON
      -DTTMLIR_ENABLE_RUNTIME=ON
    GIT_REPOSITORY git@github.com:tenstorrent/tt-mlir.git
    GIT_TAG 97fdf8d6b62d6dd5f66d2d2e0684cf8d9c7a81f5
    GIT_PROGRESS ON
    INSTALL_COMMAND ""
)

set_target_properties(tt-mlir PROPERTIES EXCLUDE_FROM_ALL TRUE)

# Link to all TTMLIR libraries in ${TTMLIR_INSTALL_DIR}/install/lib
set(TTMLIR_LIB_DIR ${TTMLIR_LIB_DIR} PARENT_SCOPE)
message(STATUS "Linking to TTMLIR libraries in ${TTMLIR_LIB_DIR}")
file(GLOB TTMLIR_LIBRARIES "${TTMLIR_LIB_DIR}/*.so")
foreach(TTMLIR_LIBRARY ${TTMLIR_LIBRARIES})
    get_filename_component(lib_name ${TTMLIR_LIBRARY} NAME_WE)
    string(REPLACE "lib" "" lib_name ${lib_name}) # Remove the "lib" prefix if it exists
    add_library(${lib_name} SHARED IMPORTED GLOBAL)
    set_target_properties(${lib_name} PROPERTIES EXCLUDE_FROM_ALL TRUE IMPORTED_LOCATION ${TTMLIR_LIBRARY})
    add_dependencies(${lib_name} tt-mlir)
endforeach()

set(FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-undef -Wno-unsafe-buffer-usage -Wno-disabled-macro-expansion")
ExternalProject_Add(
    loguru
    PREFIX ${TTPJRT_SOURCE_DIR}/third_party/loguru
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_C_COMPILER=clang-17
      -DCMAKE_CXX_COMPILER=clang++-17
      -DCMAKE_CXX_FLAGS=${FLAGS}
      -DCMAKE_INSTALL_PREFIX=${TTPJRT_SOURCE_DIR}/third_party/loguru/src/loguru-install
    GIT_REPOSITORY git@github.com:emilk/loguru.git
    GIT_TAG 4adaa185883e3c04da25913579c451d3c32cfac1
    GIT_PROGRESS ON
)
