project(cTorch_lib DESCRIPTION "cTorch lib" LANGUAGES C)

set(cTorch_lib_VERSION 0.1.0)

################################################################################
#
# Build options
#

set(TARGET_CPU_ARCH "x86" CACHE STRING
    "Target cpu architecture. \
     Supported architectures: [x86, arm] "
)

set(BACKEND_OPENBLAS_ENABLE ON CACHE BOOL
    "Build backend with openBLAS"
)

set(BACKEND_MKL_ENABLE ON CACHE BOOL
    "Build backend with Intel MKL"
)
set(BACKEND_MKL_LIB_DIR "/opt/intel/mkl/lib/intel64" CACHE PATH
    "MKL library dir. Default: /opt/intel/lib/intel64"
)
set(BACKEND_MKL_INCLUDE_DIR "/opt/intel/mkl/include" CACHE PATH
    "MKL include dir. Default: /opt/intel/include"
)

set(BACKEND_CUDA_ENABLE ON CACHE BOOL
    "Build backend with CUDA"
)

set(BACKEND_ACCELERATE_ENABLE ON CACHE BOOL
    "Build backend with Apple Accelerate"
)

set(PTHREAD_ENABLE ON CACHE BOOL
    "Build with pthread support. Default ON."
)

set(DEBUG_TEST OFF CACHE BOOL
    "Enable debug test options. Default ON. Option should only be enabled for running tests."
)

################################################################################
#
# Debug test setup
#

if(${DEBUG_TEST})
    message(
        WARNING
        "[cTorch]: DEBUG_TEST is set to ON. "
        "This opiton should only be enabled for debugging cTorch."
        "Turning it on will lead to performance regression."
    )
    # Enable debug code compiling
    add_compile_definitions(CTH_TEST_DEBUG)

    # profiling enable on linux
    if(
        ${CMAKE_SYSTEM_NAME} STREQUAL "Linux"
        AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
    )
        message("[cTorch] Append flag: -pg --coverage")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg --coverage")
    endif()
endif()

################################################################################
#
# pthread
#

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
find_package(Threads REQUIRED)

add_compile_definitions(USE_PTHREAD)

################################################################################
#
# Source files
#

set(core_src
    bit_array.c
    config.c
    consts.c
    engine.c
    graph.c
    logger_util.c
    mem_util.c
    node.c
    operator.c
    params.c
    pool.c
    queue.c
    scheduler.c
    sharder.c
    storage.c
)

add_subdirectory(operators) # ${operator_src}

if(${DEBUG_TEST})
    set(core_src
        ${core_src}
        debug_util.c
    )
endif()

# asprintf
add_compile_definitions(_GNU_SOURCE)

################################################################################
#
# Setup cTorch target
#

add_library(cTorch
    STATIC
    ${core_src}
    ${operator_src}
)
target_include_directories(cTorch PUBLIC ${cTorch_project_SOURCE_DIR})
set_target_properties(cTorch PROPERTIES C_STANDARD 11)
set_target_properties(cTorch PROPERTIES VERSION ${cTorch_lib_VERSION})

# link pthread
if(PTHREAD_ENABLE)
    target_link_libraries(cTorch PUBLIC Threads::Threads)
endif()

# link gcov for test on linux
if(
    ${DEBUG_TEST}
    AND  ${CMAKE_SYSTEM_NAME} STREQUAL "Linux"
    AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
)
    message("[cTorch]: Will link gcov")
    target_link_libraries(cTorch PUBLIC gcov)
endif()


################################################################################
#
# Backends define & link
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
endif()

# MKL
if(${BACKEND_MKL_ENABLE})
    add_compile_definitions(BACKEND_MKL)
    find_library(
        MKL_CORE_LIB
        NAMES mkl_core
        PATHS ${BACKEND_MKL_LIB_DIR}
        REQUIRED
    )
    find_library(
        MKL_INTERFACE_LIB
        NAMES mkl_intel_ilp64
        PATHS ${BACKEND_MKL_LIB_DIR}
        REQUIRED
    )
    find_library(
        MKL_THREAD_LIB
        NAMES mkl_gnu_thread
        PATHS ${BACKEND_MKL_LIB_DIR}
        REQUIRED
    )
    find_package(OpenMP REQUIRED)
    set(${CMAKE_C_FLAGS} "${CMAKE_C_FLAGS} -m64")
    target_include_directories(cTorch PRIVATE ${BACKEND_MKL_INCLUDE_DIR})
    # !Link order matters!
    target_link_libraries(
        cTorch PUBLIC
        ${MKL_INTERFACE_LIB}
        ${MKL_CORE_LIB}
        ${MKL_THREAD_LIB}
        OpenMP::OpenMP_C
    )
endif()

# CUDA
if(${BACKEND_CUDA_ENABLE})
    add_compile_definitions(BACKEND_CUDA)
endif()

# Apple
if(${BACKEND_ACCELERATE_ENABLE})
    add_compile_definitions(BACKEND_ACCELERATE)
endif()


################################################################################
#
# Install config
#

include(GNUInstallDirs)
install(TARGETS cTorch
    EXPORT cTorchTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/cTorch
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}/cTorch
)
install(
    DIRECTORY ${cTorch_lib_SOURCE_DIR}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

################################################################################
#
# Setup CMake package configures
#

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
  "${PROJECT_SOURCE_DIR}/CMake/cTorchConfigVersion.cmake"
  VERSION ${cTorch_lib_VERSION}
  COMPATIBILITY AnyNewerVersion
)
export(EXPORT cTorchTargets
  FILE "${PROJECT_SOURCE_DIR}/CMake/cTorchTargets.cmake"
)
configure_file(
    "${PROJECT_SOURCE_DIR}/CMake/cTorchTargets.cmake"
    "${PROJECT_SOURCE_DIR}/CMake/cTorchConfig.cmake"
     COPYONLY
)
