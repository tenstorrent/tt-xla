include(ExternalProject)
add_subdirectory(pjrt_c_api)

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
