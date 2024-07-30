include(ExternalProject)
add_subdirectory(pjrt_c_api)

set(TTMLIR_LIB_DIR ${PROJECT_SOURCE_DIR}/third_party/tt-mlir/src/tt-mlir-build/lib/SharedLib)
ExternalProject_Add(
    tt-mlir
    PREFIX ${TTPJRT_SOURCE_DIR}/third_party/tt-mlir
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_C_COMPILER=clang
      -DCMAKE_CXX_COMPILER=clang++
      -DTT_RUNTIME_ENABLE_TTNN=ON
      -DTTMLIR_ENABLE_STABLEHLO=ON
      -DTTMLIR_ENABLE_RUNTIME=ON
    GIT_REPOSITORY git@github.com:tenstorrent/tt-mlir.git
    GIT_TAG 458061f6ef3f91aba23086dc4326f351390ccf9d
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
