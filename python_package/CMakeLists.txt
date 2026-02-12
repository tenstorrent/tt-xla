if(TTXLA_ENABLE_EWHEEL_INSTALL)
  # Editable installation of all python packages in the python_package directory.
  add_custom_target(install-ttxla-python-packages ALL
      COMMAND ln -sf $<TARGET_FILE:TTPJRTTTDylib> ${CMAKE_SOURCE_DIR}/python_package/pjrt_plugin_tt/pjrt_plugin_tt.so
      COMMENT "Installing pjrt_plugin_tt.so to python_package/pjrt_plugin_tt"
      COMMAND pip install -e ${CMAKE_SOURCE_DIR}/python_package --no-deps --no-build-isolation
      COMMENT "Running editable installation..."
      USES_TERMINAL
  )

  add_dependencies(install-ttxla-python-packages TTPJRTTTDylib bundle-tracy bundle-ttnn)
endif()

# Create symlinks to the original tracy and ttnn modules from tt-metal.
# These symlinks allow our wrapper modules to import and delegate to the original code.
add_custom_target(bundle-tracy ALL
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${TTMLIR_LIB_DIR}/tt-metal/tools/tracy ${CMAKE_CURRENT_SOURCE_DIR}/tracy/_original
  DEPENDS tt-mlir
)

add_custom_target(bundle-ttnn ALL
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${TTMLIR_LIB_DIR}/tt-metal/ttnn/ttnn ${CMAKE_CURRENT_SOURCE_DIR}/ttnn/_original
  DEPENDS tt-mlir
)

# Fix RPATH of _ttnn.so to find shared libraries in pjrt_plugin_tt/lib at runtime (wheel installation).
find_program(PATCHELF patchelf REQUIRED)

install(CODE "
  file(GLOB_RECURSE TTNN_SO_FILES \"${CMAKE_CURRENT_SOURCE_DIR}/build/lib.*/pjrt_plugin_tt/tt-metal/ttnn/ttnn/_ttnn.so\")
  if(TTNN_SO_FILES)
    list(GET TTNN_SO_FILES 0 TTNN_SO_SOURCE)
    message(STATUS \"Fixing RPATH for _ttnn.so: \${TTNN_SO_SOURCE}\")
    execute_process(
      COMMAND \"${PATCHELF}\" --add-rpath \"\$ORIGIN/../../pjrt_plugin_tt/lib\" \"\${TTNN_SO_SOURCE}\"
      COMMAND_ERROR_IS_FATAL ANY
    )
  else()
    message(FATAL_ERROR \"Could not find _ttnn.so to patch RPATH in ${CMAKE_CURRENT_SOURCE_DIR}/build/\")
  endif()
")
