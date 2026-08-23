# Apple Metal build for the `_quixicore_C` extension.
#
# Two artifacts, both installed into the `vllm` package:
#
#   _quixicore_C.<soabi>.so   ObjC++ pybind11 module (csrc/quixicore/tm_metal/)
#   quixicore_metal.metallib  the vendored MSL, compiled ahead of time
#
# The metallib is built here rather than at import. QuixiCore-Metal's own
# `tk_torch` JIT-compiles it on first import, which costs ~22 s cold on an
# M5 Max and needs the Xcode Metal toolchain on every serving host; a served
# fork pays that once at build time instead.

if (NOT APPLE)
  message(FATAL_ERROR "VLLM_TARGET_DEVICE=metal requires macOS.")
endif()

enable_language(OBJCXX)

find_program(XCRUN_EXECUTABLE xcrun REQUIRED)

set(QUIXICORE_METAL_DIR "${CMAKE_CURRENT_LIST_DIR}/../csrc/quixicore/metal")
set(QUIXICORE_METAL_INCLUDE "${QUIXICORE_METAL_DIR}/include/metal")

#
# Compile every vendored kernel into one metallib.
#
# GLOB rather than an explicit list: upstream ships 79 files and resyncs add
# more, and a stale hand-maintained list fails as a missing kernel at runtime
# rather than at build time. CONFIGURE_DEPENDS re-globs when the tree changes.
#
file(GLOB_RECURSE VLLM_METAL_KERNEL_SRC CONFIGURE_DEPENDS
  "${QUIXICORE_METAL_DIR}/kernels/*.metal")

if (NOT VLLM_METAL_KERNEL_SRC)
  message(FATAL_ERROR
    "No .metal sources under ${QUIXICORE_METAL_DIR}/kernels. "
    "The QuixiCore-Metal kernels are vendored; see that directory's README.")
endif()

list(LENGTH VLLM_METAL_KERNEL_SRC VLLM_METAL_KERNEL_COUNT)
message(STATUS "Metal kernels: ${VLLM_METAL_KERNEL_COUNT} sources")

# The header-only tile substrate. Every kernel includes tk.metal, which pulls
# in this whole tree, but the compiler reports none of it as a dependency --
# so without listing it here a header-only edit leaves a stale metallib in
# place and the build silently keeps running the old shaders. Upstream's own
# CMake carries the same note against the same hazard.
file(GLOB_RECURSE VLLM_METAL_SUBSTRATE CONFIGURE_DEPENDS
  "${QUIXICORE_METAL_INCLUDE}/*.metal")

set(VLLM_METALLIB "${CMAKE_CURRENT_BINARY_DIR}/quixicore_metal.metallib")

add_custom_command(
  OUTPUT ${VLLM_METALLIB}
  COMMAND ${XCRUN_EXECUTABLE} metal
          # metal4.0 for mpp::tensor_ops (M5 GPU neural accelerators) in
          # qgemm_sm_t; the pre-existing kernels compile unchanged under it
          -std=metal4.0 -O2
          -I ${QUIXICORE_METAL_INCLUDE}
          # No .metal currently includes from kernels/common -- only the host
          # header does -- but upstream's build passes it, so keep it here too
          # rather than discover the difference on a resync.
          -I ${QUIXICORE_METAL_DIR}/kernels/common
          ${VLLM_METAL_KERNEL_SRC}
          -o ${VLLM_METALLIB}
  DEPENDS ${VLLM_METAL_KERNEL_SRC} ${VLLM_METAL_SUBSTRATE}
  COMMENT "Compiling ${VLLM_METAL_KERNEL_COUNT} Metal kernels into quixicore_metal.metallib"
  VERBATIM)

add_custom_target(quixicore_metallib ALL DEPENDS ${VLLM_METALLIB})

#
# The ObjC++ binding.
#
file(GLOB VLLM_QUIXICORE_METAL_EXT_SRC CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/../csrc/quixicore/tm_metal/*.mm")

define_extension_target(
  _quixicore_C
  DESTINATION vllm
  LANGUAGE OBJCXX
  SOURCES ${VLLM_QUIXICORE_METAL_EXT_SRC}
  COMPILE_FLAGS -fobjc-arc -O3
  INCLUDE_DIRECTORIES
    "${QUIXICORE_METAL_DIR}/kernels/common"
  WITH_SOABI)

add_dependencies(_quixicore_C quixicore_metallib)

target_link_libraries(_quixicore_C PRIVATE
  "-framework Metal" "-framework Foundation" "-framework QuartzCore")

# The torch 2.13 CMake targets rewrite libc++ and libunwind to @rpath names.
# Those two libraries are OS-owned dyld-cache residents, not files in the
# torch wheel, so a Python extension has no runpath that can resolve them.
# Restore their canonical system install names after linking.
find_program(INSTALL_NAME_TOOL_EXECUTABLE install_name_tool REQUIRED)
add_custom_command(TARGET _quixicore_C POST_BUILD
  COMMAND ${INSTALL_NAME_TOOL_EXECUTABLE} -change
          @rpath/libc++.1.dylib /usr/lib/libc++.1.dylib
          $<TARGET_FILE:_quixicore_C>
  COMMAND ${INSTALL_NAME_TOOL_EXECUTABLE} -change
          @rpath/libunwind.1.dylib /usr/lib/system/libunwind.dylib
          $<TARGET_FILE:_quixicore_C>
  # Editable builds place the extension directly in the source package and
  # do not reliably run the FILES install rule below.  Put the metallib beside
  # the extension at link time as well; regular wheels harmlessly install the
  # same file a second time through the component rule.
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${VLLM_METALLIB}
          $<TARGET_FILE_DIR:_quixicore_C>/quixicore_metal.metallib
  # PEP 660 editable installs import Python and extension modules from the
  # source tree even when setuptools staged auxiliary data in build/lib.
  # Mirror the shader library into that package as part of the Metal build.
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${VLLM_METALLIB}
          ${CMAKE_CURRENT_LIST_DIR}/../vllm/quixicore_metal.metallib
  VERBATIM)

# pybind11 modules need libtorch_python, which define_extension_target does
# not link (the stable-ABI extensions must not link it).
find_library(QUIXICORE_METAL_TORCH_PYTHON_LIB torch_python
  PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)
if (NOT QUIXICORE_METAL_TORCH_PYTHON_LIB)
  message(FATAL_ERROR "libtorch_python not found for _quixicore_C")
endif()
target_link_libraries(_quixicore_C PRIVATE ${QUIXICORE_METAL_TORCH_PYTHON_LIB})

# The extension locates the metallib beside itself inside the installed
# package, so both land in vllm/.
install(FILES ${VLLM_METALLIB} DESTINATION vllm COMPONENT _quixicore_C)
