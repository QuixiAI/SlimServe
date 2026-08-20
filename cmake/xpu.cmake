# Intel XPU (SYCL / oneAPI) build for the `_quixicore_C` extension.
#
# Mirrors cmake/metal.cmake: the vendored QuixiCore-XPU tree under
# csrc/quixicore/xpu/ is compiled with icpx -fsycl into ONE shared op library,
# and a pybind11 binding (csrc/quixicore/tm_xpu/) links against it. Both land
# in the `vllm` package:
#
#   _quixicore_C.<soabi>.so          pybind11 module (queue off the torch XPU stream)
#   libquixicore_xpu_ops.so          the SYCL kernels + runtime + dispatch
#
# Two hard requirements carried over from QuixiCore-XPU/bindings/pytorch/README.md:
#   1. The op library MUST be a shared object built by icpx -fsycl. A static
#      archive linked into the extension does not self-register its SYCL device
#      images with the runtime ProgramManager -> kernel submit segfaults.
#   2. The final extension link must be a -fsycl device link. Here that is
#      guaranteed by requiring icpx as the project C++ compiler and putting
#      -fsycl on the extension's link options (setup.py passes
#      -DCMAKE_CXX_COMPILER=icpx for VLLM_TARGET_DEVICE=xpu).

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fsycl" VLLM_XPU_HAS_FSYCL)
if (NOT VLLM_XPU_HAS_FSYCL)
  message(FATAL_ERROR
    "VLLM_TARGET_DEVICE=xpu needs a SYCL compiler (icpx). "
    "source /opt/intel/oneapi/setvars.sh and configure with "
    "CMAKE_CXX_COMPILER=icpx (setup.py does this when icpx is on PATH). "
    "Current compiler: ${CMAKE_CXX_COMPILER}")
endif()

set(QUIXICORE_XPU_DIR "${CMAKE_CURRENT_LIST_DIR}/../csrc/quixicore/xpu")

# Optional AoT target. Empty = SPIR-V JIT at first launch (Battlemage/B70:
# -DVLLM_XPU_SYCL_TARGETS=bmg or "spir64_gen -device bmg"). Perf/startup lever,
# not a correctness requirement.
set(VLLM_XPU_SYCL_TARGETS "$ENV{VLLM_XPU_SYCL_TARGETS}" CACHE STRING
    "SYCL AoT device target(s) for -fsycl-targets (empty = JIT).")

# Deterministic fp32: icpx defaults to -fp-model=fast, which lets the device
# use reciprocal-multiply division and reassociate; combined with Level Zero's
# ~2.5 ULP native divide this broke QuixiCore's bit-exact codec contracts
# (TurboQuant zero-point flips). Keep upstream's flags verbatim.
set(VLLM_XPU_SYCL_FLAGS -fsycl -fp-model=precise -foffload-fp32-prec-div
    -foffload-fp32-prec-sqrt)
if (VLLM_XPU_SYCL_TARGETS)
  list(APPEND VLLM_XPU_SYCL_FLAGS -fsycl-targets=${VLLM_XPU_SYCL_TARGETS})
endif()

#
# The op library. GLOB rather than a hand list, as metal.cmake does: upstream
# resyncs add kernels, and a stale list fails at runtime instead of build time.
#
file(GLOB_RECURSE VLLM_XPU_OPS_SRC CONFIGURE_DEPENDS
  "${QUIXICORE_XPU_DIR}/src/*.cpp"
  "${QUIXICORE_XPU_DIR}/kernels/*.sycl.cpp")
if (NOT VLLM_XPU_OPS_SRC)
  message(FATAL_ERROR "No SYCL sources under ${QUIXICORE_XPU_DIR}; see its README.md.")
endif()

# oneDNN vendor variants are optional (Variant::vendor falls back to SYCL).
option(VLLM_XPU_ENABLE_ONEDNN "Build the oneDNN vendor variants" ON)
set(VLLM_XPU_HAS_ONEDNN OFF)
if (VLLM_XPU_ENABLE_ONEDNN)
  find_package(dnnl CONFIG QUIET)
  if (dnnl_FOUND)
    set(VLLM_XPU_HAS_ONEDNN ON)
    file(GLOB_RECURSE VLLM_XPU_ONEDNN_SRC CONFIGURE_DEPENDS
      "${QUIXICORE_XPU_DIR}/kernels/*.onednn.cpp")
    list(APPEND VLLM_XPU_OPS_SRC ${VLLM_XPU_ONEDNN_SRC})
    message(STATUS "QuixiCore XPU: oneDNN found; vendor variants enabled.")
  else()
    message(STATUS "QuixiCore XPU: oneDNN not found; vendor variants fall back to SYCL.")
  endif()
endif()

# Header-only substrate: the compiler does report these as dependencies for
# .cpp TUs (unlike Metal), but list them so a CONFIGURE_DEPENDS re-glob also
# fires when a header is added.
file(GLOB_RECURSE VLLM_XPU_HEADERS CONFIGURE_DEPENDS
  "${QUIXICORE_XPU_DIR}/include/*.hpp" "${QUIXICORE_XPU_DIR}/kernels/*.hpp")

list(LENGTH VLLM_XPU_OPS_SRC VLLM_XPU_OPS_COUNT)
message(STATUS "QuixiCore XPU: ${VLLM_XPU_OPS_COUNT} SYCL translation units")

add_library(quixicore_xpu_ops SHARED ${VLLM_XPU_OPS_SRC} ${VLLM_XPU_HEADERS})
target_compile_features(quixicore_xpu_ops PUBLIC cxx_std_20)
target_include_directories(quixicore_xpu_ops
  PUBLIC "${QUIXICORE_XPU_DIR}/include"
  PRIVATE "${QUIXICORE_XPU_DIR}/kernels")
target_compile_options(quixicore_xpu_ops PUBLIC ${VLLM_XPU_SYCL_FLAGS})
target_link_options(quixicore_xpu_ops PUBLIC ${VLLM_XPU_SYCL_FLAGS})
target_compile_options(quixicore_xpu_ops PRIVATE -O3)
if (VLLM_XPU_HAS_ONEDNN)
  target_compile_definitions(quixicore_xpu_ops PUBLIC QUIXICORE_XPU_HAS_ONEDNN)
  target_link_libraries(quixicore_xpu_ops PUBLIC DNNL::dnnl)
endif()
find_library(VLLM_XPU_ZE_LOADER ze_loader)
if (VLLM_XPU_ZE_LOADER)
  target_compile_definitions(quixicore_xpu_ops PRIVATE QUIXICORE_XPU_HAS_LEVEL_ZERO)
  target_link_libraries(quixicore_xpu_ops PUBLIC ${VLLM_XPU_ZE_LOADER})
  message(STATUS "QuixiCore XPU: Level Zero loader found; IPC enabled.")
else()
  message(STATUS "QuixiCore XPU: Level Zero loader not found; IPC stubs.")
endif()
#
# ONE SYCL runtime per process. torch+xpu wheels bundle their own libsycl.so.9 /
# libur_* (pip intel-sycl-rt) next to site-packages, and libtorch_xpu resolves
# them by RUNPATH. If our library instead resolves /opt/intel/oneapi's copy
# (which `source setvars.sh` puts on LD_LIBRARY_PATH), two SYCL runtimes load
# and the first XPU allocation segfaults (measured 2026-08-18 on QuadB70).
# Pin ours to torch's runtime dir with DT_RPATH (--disable-new-dtags), which
# outranks LD_LIBRARY_PATH; the compiler's own dir is the fallback for
# non-torch environments. oneDNN's dir rides along so setvars is not needed
# at import time either.
#
find_library(VLLM_XPU_TORCH_SYCL_LIB sycl
  NAMES libsycl.so.9 sycl
  HINTS "${TORCH_INSTALL_PREFIX}/../../.." "${TORCH_INSTALL_PREFIX}/lib"
  NO_DEFAULT_PATH)
if (VLLM_XPU_TORCH_SYCL_LIB)
  get_filename_component(VLLM_XPU_SYCL_RUNTIME_DIR "${VLLM_XPU_TORCH_SYCL_LIB}" DIRECTORY)
  message(STATUS "QuixiCore XPU: SYCL runtime pinned to torch's ${VLLM_XPU_SYCL_RUNTIME_DIR}")
else()
  set(VLLM_XPU_SYCL_RUNTIME_DIR "")
  message(WARNING "QuixiCore XPU: torch does not bundle libsycl; the compiler's runtime will be used. Do not mix with a torch+xpu wheel that does.")
endif()
set(VLLM_XPU_RPATH "\$ORIGIN")
if (VLLM_XPU_SYCL_RUNTIME_DIR)
  list(APPEND VLLM_XPU_RPATH "${VLLM_XPU_SYCL_RUNTIME_DIR}")
endif()
if (VLLM_XPU_HAS_ONEDNN)
  get_target_property(VLLM_XPU_DNNL_LOC DNNL::dnnl LOCATION)
  get_filename_component(VLLM_XPU_DNNL_DIR "${VLLM_XPU_DNNL_LOC}" DIRECTORY)
  list(APPEND VLLM_XPU_RPATH "${VLLM_XPU_DNNL_DIR}")
endif()
list(JOIN VLLM_XPU_RPATH ":" VLLM_XPU_RPATH_STR)

set_target_properties(quixicore_xpu_ops PROPERTIES
  OUTPUT_NAME quixicore_xpu_ops
  BUILD_RPATH "${VLLM_XPU_RPATH_STR}"
  INSTALL_RPATH "${VLLM_XPU_RPATH_STR}"
  BUILD_WITH_INSTALL_RPATH ON
  BUILD_RPATH_USE_ORIGIN ON)
target_link_options(quixicore_xpu_ops PRIVATE -Wl,--disable-new-dtags)
if (VLLM_XPU_SYCL_RUNTIME_DIR)
  target_link_directories(quixicore_xpu_ops PUBLIC "${VLLM_XPU_SYCL_RUNTIME_DIR}")
endif()

#
# The pybind11 binding. The source keeps the upstream `.sycl` suffix so it
# diffs cleanly against QuixiCore-XPU/bindings/pytorch/tk_xpu_ext.sycl; tell
# CMake it is C++ (icpx compiles it with -fsycl through the ops PUBLIC flags).
#
file(GLOB VLLM_QUIXICORE_XPU_EXT_SRC CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_LIST_DIR}/../csrc/quixicore/tm_xpu/*.sycl"
  "${CMAKE_CURRENT_LIST_DIR}/../csrc/quixicore/tm_xpu/*.cpp")
set_source_files_properties(${VLLM_QUIXICORE_XPU_EXT_SRC} PROPERTIES LANGUAGE CXX)

define_extension_target(
  _quixicore_C
  DESTINATION vllm
  LANGUAGE CXX
  SOURCES ${VLLM_QUIXICORE_XPU_EXT_SRC}
  COMPILE_FLAGS -O3
  INCLUDE_DIRECTORIES "${QUIXICORE_XPU_DIR}/include"
  LIBRARIES quixicore_xpu_ops
  WITH_SOABI)

set_target_properties(_quixicore_C PROPERTIES
  BUILD_RPATH "${VLLM_XPU_RPATH_STR}"
  INSTALL_RPATH "${VLLM_XPU_RPATH_STR}"
  BUILD_WITH_INSTALL_RPATH ON
  BUILD_RPATH_USE_ORIGIN ON)
target_link_options(_quixicore_C PRIVATE -Wl,--disable-new-dtags)

# pybind11 modules need libtorch_python (stable-ABI extensions must not link it).
find_library(QUIXICORE_XPU_TORCH_PYTHON_LIB torch_python
  PATHS "${TORCH_INSTALL_PREFIX}/lib" NO_DEFAULT_PATH)
if (NOT QUIXICORE_XPU_TORCH_PYTHON_LIB)
  message(FATAL_ERROR "libtorch_python not found for _quixicore_C")
endif()
target_link_libraries(_quixicore_C PRIVATE ${QUIXICORE_XPU_TORCH_PYTHON_LIB})

# Editable installs import from the source tree; put the op library beside the
# extension at link time and mirror it into vllm/, as metal.cmake does with the
# metallib. Regular wheels also get it through the component install below.
add_custom_command(TARGET _quixicore_C POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:quixicore_xpu_ops>
          $<TARGET_FILE_DIR:_quixicore_C>/$<TARGET_FILE_NAME:quixicore_xpu_ops>
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:quixicore_xpu_ops>
          ${CMAKE_CURRENT_LIST_DIR}/../vllm/$<TARGET_FILE_NAME:quixicore_xpu_ops>
  VERBATIM)

install(TARGETS quixicore_xpu_ops LIBRARY DESTINATION vllm COMPONENT _quixicore_C)
