cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(cTorch_project DESCRIPTION "cTorch project" LANGUAGES C)

set(CMAKE_C_STANDARD 11)

################################################################################
#
# Build options
#

set(TARGET_CPU_ARCH "x86" CACHE STRING
    "Target cpu architecture. \
     Supported architectures: [x86, arm] "
)

set(BACKEND_OPENBLAS_ENABLE OFF CACHE BOOL
    "Build with backend openBLAS backend"
)

set(BACKEND_MKL_ENABLE OFF CACHE BOOL
    "Build backend with Intel MKL backend"
)
set(BACKEND_MKL_LIB_DIR "/opt/intel/mkl/lib/intel64" CACHE PATH
    "MKL library dir. Default: /opt/intel/lib/intel64"
)
set(BACKEND_MKL_INCLUDE_DIR "/opt/intel/mkl/include" CACHE PATH
    "MKL include dir. Default: /opt/intel/include"
)

set(BACKEND_CUDA_ENABLE OFF CACHE BOOL
    "Build with CUDA backend"
)

set(BACKEND_APPLE_ENABLE OFF CACHE BOOL
    "Build with Apple backend"
)

set(DEBUG_TEST OFF CACHE BOOL
    "Enable debug test options. Default OFF. Option should only be enabled for running tests."
)


################################################################################
#
# Backends compiling enable
#

if(${TARGET_CPU_ARCH} STREQUAL "x86")
    add_compile_definitions(BACKEND_CPU_X86)
elseif(${TARGET_CPU_ARCH} STREQUAL "arm")
    add_compile_definitions(BACKEND_CPU_ARM)
else()
    message(FATAL_ERROR "Unsupported CPU architecture: " ${TARGET_CPU_ARCH})
endif()

# Openblas
if(${BACKEND_OPENBLAS_ENABLE})
    add_compile_definitions(BACKEND_OPENBLAS)
    message("[cTorch]: Enable building with OpenBLAS backend")
endif()

# MKL
if(${BACKEND_MKL_ENABLE})
    add_compile_definitions(BACKEND_MKL)
    message("[cTorch]: Enable building with Intel MKL backend")
endif()

# CUDA
if(${BACKEND_CUDA_ENABLE})
    enable_language(CUDA)
    add_compile_definitions(BACKEND_CUDA)
    message("[cTorch]: Enable building with CUDA backend")
endif()

# Apple
if(${BACKEND_APPLE_ENABLE})
    add_compile_definitions(BACKEND_APPLE)
    message("[cTorch]: Enable building with Apple backend")
endif()

# asprintf
add_compile_definitions(_GNU_SOURCE)

################################################################################
#
# Debug test macro
#

if(${DEBUG_TEST})
    message(
        WARNING
        "[cTorch]: DEBUG_TEST is turned ON. "
        "This opiton should only be enabled for debugging cTorch. "
        "Turning it on will lead to performance regression."
    )
    # Enable debug code compiling
    add_compile_definitions(CTH_TEST_DEBUG)
endif()


################################################################################
#
# cTorch lib
#
add_subdirectory(cTorch)

################################################################################
#
# Tests
#
if(${DEBUG_TEST})
    enable_language(CXX)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)
    add_subdirectory(third_party/googletest)
    add_subdirectory(tests)
endif()
