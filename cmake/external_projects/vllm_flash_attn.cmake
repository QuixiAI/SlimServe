# vLLM flash attention requires VLLM_GPU_ARCHES to contain the set of target
# arches in the CMake syntax (75-real, 89-virtual, etc), since we clear the
# arches in the CUDA case (and instead set the gencodes on a per file basis)
# we need to manually set VLLM_GPU_ARCHES here.
if(VLLM_GPU_LANG STREQUAL "CUDA")
  foreach(_ARCH ${CUDA_ARCHS})
    string(REPLACE "." "" _ARCH "${_ARCH}")
    list(APPEND VLLM_GPU_ARCHES "${_ARCH}-real")
  endforeach()
endif()

# Upstream vllm-flash-attn hardcodes the FA2 kernels (flash_fwd_*.cu) to
# "8.0+PTX". For an SM12x build its loose intersection therefore picks 8.0+PTX,
# so the RTX 50xx / DGX Spark extension holds only sm_80 SASS plus compute_80
# PTX, and the driver has to JIT that PTX at first launch. With a toolkit newer
# than the driver (CUDA 13.3 build on an R595/13.2 driver) the JIT fails with
# cudaErrorUnsupportedPtxVersion ("the provided PTX was compiled with an
# unsupported toolchain"), the same failure MARLIN_OTHER_ARCHS and
# HADACORE_ARCHS had. Add SM12x SASS to the FA2 kernels, mirroring those lists.
#
# The arch list is resolved here, with SlimServe's cuda_archs_loose_intersection
# (it understands the a/f suffixes; the copy vllm-flash-attn installs over it
# does not), and applied to the FA2 sources after FetchContent_MakeAvailable.
# FA3 is Hopper-only (9.0a) and is left alone.
set(FA2_SM12X_ARCHS)
if(VLLM_GPU_LANG STREQUAL "CUDA")
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 13.0)
    cuda_archs_loose_intersection(FA2_SM12X_ARCHS "12.0f" "${CUDA_ARCHS}")
  elseif(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL 12.8)
    cuda_archs_loose_intersection(FA2_SM12X_ARCHS "12.0a;12.1a" "${CUDA_ARCHS}")
  endif()
endif()

#
# Build vLLM flash attention from source
#
# IMPORTANT: This has to be the last thing we do, because vllm-flash-attn uses the same macros/functions as vLLM.
# Because functions all belong to the global scope, vllm-flash-attn's functions overwrite vLLMs.
# They should be identical but if they aren't, this is a massive footgun.
#
# The vllm-flash-attn install rules are nested under vllm to make sure the library gets installed in the correct place.
# To only install vllm-flash-attn, use --component _vllm_fa2_C (for FA2), --component _vllm_fa3_C (for FA3),
# or --component _vllm_fa4_cutedsl_C (for FA4 CuteDSL Python files).
# If no component is specified, vllm-flash-attn is still installed.

# If VLLM_FLASH_ATTN_SRC_DIR is set, vllm-flash-attn is installed from that directory instead of downloading.
# This is to enable local development of vllm-flash-attn within vLLM.
# It can be set as an environment variable or passed as a cmake argument.
# The environment variable takes precedence.
if (DEFINED ENV{VLLM_FLASH_ATTN_SRC_DIR})
  set(VLLM_FLASH_ATTN_SRC_DIR $ENV{VLLM_FLASH_ATTN_SRC_DIR})
endif()

if(VLLM_FLASH_ATTN_SRC_DIR)
  FetchContent_Declare(
          vllm-flash-attn SOURCE_DIR
          ${VLLM_FLASH_ATTN_SRC_DIR}
          BINARY_DIR ${CMAKE_BINARY_DIR}/vllm-flash-attn
  )
else()
  FetchContent_Declare(
          vllm-flash-attn
          GIT_REPOSITORY https://github.com/vllm-project/flash-attention.git
          GIT_TAG ed4b7342bc8f0489dd9b649d5288867e35fc6a32
          GIT_PROGRESS TRUE
          # Don't share the vllm-flash-attn build between build types
          BINARY_DIR ${CMAKE_BINARY_DIR}/vllm-flash-attn
  )
endif()

# Make sure vllm-flash-attn install rules are nested under vllm/
# ALL_COMPONENTS ensures the save/modify/restore runs exactly once regardless
# of how many components are being installed, avoiding double-append of /vllm/.
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY FALSE)" ALL_COMPONENTS)
install(CODE "set(OLD_CMAKE_INSTALL_PREFIX \"\${CMAKE_INSTALL_PREFIX}\")" ALL_COMPONENTS)
install(CODE "set(CMAKE_INSTALL_PREFIX \"\${CMAKE_INSTALL_PREFIX}/vllm/\")" ALL_COMPONENTS)

# Fetch the vllm-flash-attn library
FetchContent_MakeAvailable(vllm-flash-attn)
message(STATUS "vllm-flash-attn is available at ${vllm-flash-attn_SOURCE_DIR}")

# Apply FA2_SM12X_ARCHS (see above) to the FA2 kernels. Source-file properties
# are directory scoped, so they are set in vllm-flash-attn's directory, where
# its own set_gencode_flags_for_srcs already appended the 8.0+PTX gencodes.
if(FA2_SM12X_ARCHS AND TARGET _vllm_fa2_C)
  file(GLOB _FA2_GEN_SRCS
    "${vllm-flash-attn_SOURCE_DIR}/csrc/flash_attn/src/flash_fwd_*.cu")
  if(NOT _FA2_GEN_SRCS)
    message(FATAL_ERROR "No FA2 kernel sources found under "
      "${vllm-flash-attn_SOURCE_DIR}/csrc/flash_attn/src; the SM12x arch "
      "override in vllm_flash_attn.cmake needs updating.")
  endif()
  foreach(_ARCH ${FA2_SM12X_ARCHS})
    string(REPLACE "." "" _ARCH "${_ARCH}")
    # Same flag shape as set_gencode_flag_for_srcs in cmake/utils.cmake.
    set(_FLAG -gencode arch=compute_${_ARCH},code=sm_${_ARCH})
    set_property(
      SOURCE ${_FA2_GEN_SRCS}
      DIRECTORY "${vllm-flash-attn_SOURCE_DIR}"
      APPEND PROPERTY
      COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:${_FLAG}>")
  endforeach()
  message(STATUS "FA2 SM12x archs: ${FA2_SM12X_ARCHS}")
endif()

# Restore the install prefix after FA's install rules
install(CODE "set(CMAKE_INSTALL_PREFIX \"\${OLD_CMAKE_INSTALL_PREFIX}\")" ALL_COMPONENTS)
install(CODE "set(CMAKE_INSTALL_LOCAL_ONLY TRUE)" ALL_COMPONENTS)

# Install shared Python files for both FA2 and FA3 components
foreach(_FA_COMPONENT _vllm_fa2_C _vllm_fa3_C)
  # Ensure the vllm/vllm_flash_attn directory exists before installation
  install(CODE "file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn\")"
    COMPONENT ${_FA_COMPONENT})

  # Copy vllm_flash_attn python files (except __init__.py and flash_attn_interface.py
  # which are source-controlled in vllm)
  install(
    DIRECTORY ${vllm-flash-attn_SOURCE_DIR}/vllm_flash_attn/
    DESTINATION vllm/vllm_flash_attn
    COMPONENT ${_FA_COMPONENT}
    FILES_MATCHING PATTERN "*.py"
    PATTERN "__init__.py" EXCLUDE
    PATTERN "flash_attn_interface.py" EXCLUDE
  )

endforeach()

#
# FA4 CuteDSL component
# This is a Python-only component that copies the flash_attn/cute directory
# and transforms imports to match our package structure.
#
add_custom_target(_vllm_fa4_cutedsl_C)

# Install flash_attn/cute directory (needed for FA4).
# When using a local source dir (VLLM_FLASH_ATTN_SRC_DIR), create a symlink
# so edits to cute-dsl Python files take effect immediately without rebuilding.
# Otherwise, copy files and transform flash_attn.cute imports to
# vllm.vllm_flash_attn.cute to match our package structure.
if(VLLM_FLASH_ATTN_SRC_DIR)
  install(CODE "
    set(LINK_TARGET \"${vllm-flash-attn_SOURCE_DIR}/flash_attn/cute\")
    set(LINK_NAME \"\${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn/cute\")
    file(MAKE_DIRECTORY \"\${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn\")
    file(REMOVE_RECURSE \"\${LINK_NAME}\")
    file(CREATE_LINK \"\${LINK_TARGET}\" \"\${LINK_NAME}\" SYMBOLIC)
  " COMPONENT _vllm_fa4_cutedsl_C)
else()
  install(CODE "
    file(GLOB_RECURSE CUTE_PY_FILES \"${vllm-flash-attn_SOURCE_DIR}/flash_attn/cute/*.py\")
    foreach(SRC_FILE \${CUTE_PY_FILES})
      file(RELATIVE_PATH REL_PATH \"${vllm-flash-attn_SOURCE_DIR}/flash_attn/cute\" \${SRC_FILE})
      set(DST_FILE \"\${CMAKE_INSTALL_PREFIX}/vllm/vllm_flash_attn/cute/\${REL_PATH}\")
      get_filename_component(DST_DIR \${DST_FILE} DIRECTORY)
      file(MAKE_DIRECTORY \${DST_DIR})
      file(READ \${SRC_FILE} FILE_CONTENTS)
      string(REPLACE \"flash_attn.cute\" \"vllm.vllm_flash_attn.cute\" FILE_CONTENTS \"\${FILE_CONTENTS}\")
      file(WRITE \${DST_FILE} \"\${FILE_CONTENTS}\")
    endforeach()
  " COMPONENT _vllm_fa4_cutedsl_C)
endif()
