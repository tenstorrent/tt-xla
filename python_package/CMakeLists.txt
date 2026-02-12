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

add_custom_target(patch-ttnn-rpath ALL
  COMMAND ${CMAKE_COMMAND} -E echo "Searching for _ttnn.so to patch..."
  COMMAND bash -c "find ${CMAKE_CURRENT_SOURCE_DIR}/build -name '_ttnn.so' -path '*/pjrt_plugin_tt/tt-metal/ttnn/ttnn/_ttnn.so' -exec echo 'Found: {}' \\; -exec ${PATCHELF} --add-rpath '$ORIGIN/../../pjrt_plugin_tt/lib' {} \\; || echo 'Warning: _ttnn.so not found yet'"
  COMMENT "Patching RPATH for _ttnn.so"
  VERBATIM
)
