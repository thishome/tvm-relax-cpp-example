# =============================================================================
# CMake Toolchain File: Cross-compile for NVIDIA Thor (aarch64) on x86_64
# Target SoC  : NVIDIA Thor (Orin-next, Blackwell GPU, aarch64)
# Host        : x86_64 Linux
# Compiler    : aarch64-linux-gnu-gcc/g++ (or clang cross)
# CUDA        : nvcc with --target-dir / --sysroot pointing to Thor sysroot
# =============================================================================

set(CMAKE_SYSTEM_NAME      Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# ---------------------------------------------------------------------------
# 1. Sysroot – set THOR_SYSROOT env or pass -DTHOR_SYSROOT=<path>
#    Typical location when using DRIVE AGX SDK / Jetson cross-tool:
#      /usr/aarch64-linux-gnu   (bare cross-tool, no sysroot)
#      $SDKROOT/targetfs        (full Drive/Jetson SDK sysroot)
# ---------------------------------------------------------------------------
# message(ENV{THOR_SYSROOT})
if(DEFINED ENV{THOR_SYSROOT})
    set(THOR_SYSROOT "$ENV{THOR_SYSROOT}" CACHE PATH "Thor target sysroot")
else()
    # set(THOR_SYSROOT "/usr/aarch64-linux-gnu" CACHE PATH "Thor target sysroot")
endif()

# ---------------------------------------------------------------------------
# 2. Cross C/C++ compilers
# ---------------------------------------------------------------------------
set(CROSS_COMPILE_PREFIX "aarch64-linux-gnu-" CACHE STRING "Cross-compile prefix")

find_program(CMAKE_C_COMPILER   ${CROSS_COMPILE_PREFIX}gcc   REQUIRED)
find_program(CMAKE_CXX_COMPILER ${CROSS_COMPILE_PREFIX}g++   REQUIRED)
find_program(CMAKE_ASM_COMPILER ${CROSS_COMPILE_PREFIX}as)

# ---------------------------------------------------------------------------
# 3. CUDA cross-compilation settings
#    nvcc is the HOST tool; it invokes the device compiler internally.
#    We tell nvcc to use the aarch64 host compiler via -ccbin.
# ---------------------------------------------------------------------------
if(DEFINED ENV{CUDA_PATH})
    set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDA_PATH}" CACHE PATH "CUDA root")
else()
    # Default CUDA cross-compile install location from NVIDIA SDK Manager
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda" CACHE PATH "CUDA root")
endif()

set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE FILEPATH "nvcc")

# Thor contains a Blackwell (GB10x) GPU – sm_100 / sm_101
# Also keep sm_87 (Orin Ampere) for compatibility testing
set(CMAKE_CUDA_ARCHITECTURES "87;100" CACHE STRING "CUDA architectures for Thor")

# Tell nvcc to cross-compile: use aarch64 host compiler
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE FILEPATH "CUDA host compiler")

# ---------------------------------------------------------------------------
# 4. Sysroot & search paths
# ---------------------------------------------------------------------------
set(CMAKE_SYSROOT "${THOR_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH
    "${THOR_SYSROOT}"
    "${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux"
)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)   # host tools
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)    # target libs
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)    # target headers
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# ---------------------------------------------------------------------------
# 5. Extra CUDA flags for cross-compilation
# ---------------------------------------------------------------------------
set(CUDA_CROSS_FLAGS
    "--target-dir=${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux"
    "-ccbin=${CMAKE_CXX_COMPILER}"
    "--allow-unsupported-compiler"  # remove if your gcc version is officially supported
    CACHE STRING "Extra nvcc flags for cross-compilation")
